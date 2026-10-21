"""Persistent, isolated crew-only scratch scenarios for the existing board."""
from __future__ import annotations
import copy, json, re
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

STORE = Path(__file__).resolve().parent / 'scratch_board_data'
LEGACY_STORE = Path(__file__).resolve().parent / 'output' / 'scratch_boards'

def _safe(name):
    name = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(name).strip())[:64]
    return name or 'scratch'

def names():
    return sorted({p.stem for folder in (STORE, LEGACY_STORE) if folder.exists() for p in folder.glob('*.json')})

def read(name):
    p = STORE / (_safe(name)+'.json')
    if not p.exists():
        p = LEGACY_STORE / (_safe(name)+'.json')
    data=json.loads(p.read_text(encoding='utf-8'))
    # Old scenarios remain readable; their next save migrates to scratch_board_data.
    return data

def write(name, solution, history=None, validation=None):
    STORE.mkdir(parents=True, exist_ok=True)
    p=STORE / (_safe(name)+'.json')
    data={'schema_version':2,'name':_safe(name),'solution':solution,
          'crew_assignments':assignments(solution), 'history':history or [], 'validation':validation,
          'updated_at':datetime.now().isoformat(timespec='seconds')}
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    tmp.replace(p)

def create(name, source):
    if _safe(name) in names():
        raise ValueError('A scratch board with this name already exists')
    write(name,copy.deepcopy(source))

def delete(name):
    for folder in (STORE, LEGACY_STORE):
        (folder/(_safe(name)+'.json')).unlink(missing_ok=True)

def assignments(solution):
    result={}
    for reg,moves in (solution.get('aircraft_movements') or {}).items():
        for m in moves:
            if str(m.get('type','')).upper()=='MISSION':
                mid=str(m.get('mission',''))
                if mid in result:
                    raise ValueError(f'Duplicate mission {mid} in aircraft movements')
                result[mid]={'captain':str(m.get('captain','')),'fo':str(m.get('fo','')),'aircraft':reg}
    return result

def eligible_crew(solution, mission_id, missions, pilots, availability):
    """Return assignable CAPTs/FOs, excluding blocked days and duty overlaps.

    The incumbent on the selected mission is exempted from its own assignment.
    Duty intervals include report/release buffers from the optimizer.
    """
    import optimizer as opt
    by_id={m.id:m for m in missions}
    target=by_id.get(str(mission_id))
    if target is None:
        raise ValueError(f'Unknown mission: {mission_id}')
    assigned=assignments(solution)
    occupied={}
    for mid,crew in assigned.items():
        if mid==str(mission_id) or mid not in by_id:
            continue
        for pid in (crew['captain'],crew['fo']):
            occupied.setdefault(pid,[]).append(by_id[mid])
    result={'CAPT':[], 'FO':[]}
    blocked={}
    for pid,p in sorted(pilots.items()):
        if p.role not in result:
            continue
        reason=None
        if not opt.available(pid,target,availability):
            reason='Day off / leave / unavailable'
        elif any(opt.overlap(opt.duty_interval(target),opt.duty_interval(other)) for other in occupied.get(pid,[])):
            reason='Overlapping duty assignment'
        if reason:
            blocked[pid]=reason
        else:
            result[p.role].append(pid)
    return result,blocked


def verify_crew_change(solution, mission_id, captain, fo, missions, pilots, availability):
    eligible,_=eligible_crew(solution,mission_id,missions,pilots,availability)
    if captain and captain not in eligible['CAPT']:
        raise ValueError(f'Captain {captain} is not available for this mission.')
    if fo and fo not in eligible['FO']:
        raise ValueError(f'First Officer {fo} is not available for this mission.')
    if captain and fo and not (pilots[captain].experienced or pilots[fo].experienced):
        raise ValueError('At least one crew member must be Experienced.')


def edit(solution, mission_id, captain, fo):
    new=copy.deepcopy(solution)
    hits=0
    for moves in (new.get('aircraft_movements') or {}).values():
        for m in moves:
            if m.get('type')=='MISSION' and str(m.get('mission'))==str(mission_id):
                m['captain'],m['fo']=captain,fo
                hits+=1
    if hits!=1:
        raise ValueError(f'Expected exactly one mission {mission_id}; found {hits}')
    # Legacy crew_actions are not an authoritative source for a manually edited
    # board. Recalculate swap events directly from movements in the UI.
    # Non-mission positioning rows may carry stale attached crew; preserve their
    # original data but do not treat them as scratch mission assignments.
    new['crew_actions']=[]
    new['pilot_ftl_audit']={}
    new['pilot_assignments']={}
    new['scratch_validation_status']='NOT VALIDATED'
    new['scratch_modified']=True
    return new

def validate(solution):
    import optimizer as opt
    missions,pilots,avail=opt.load_data()
    idx={m.id:m for m in missions}
    mapping=assignments(solution)
    issues=[]
    def issue(pilot,rule,detail,mid=''):
        issues.append({'Pilot':pilot,'Rule':rule,'Details':detail,'Mission':mid})
    missing=sorted(set(idx)-set(mapping))
    extra=sorted(set(mapping)-set(idx))
    for mid in missing: issue('—','Mission coverage','Missing crew assignment',mid)
    for mid in extra: issue('—','Mission coverage','Mission absent from active missions.csv',mid)
    for mid,crew in mapping.items():
        if mid not in idx: continue
        m=idx[mid]
        if str(m.fixed_aircraft).replace('-','').upper()!=str(crew['aircraft']).replace('-','').upper():
            issue('—','Aircraft assignment','Different from fixed aircraft',mid)
        c,f=crew['captain'],crew['fo']
        if not c: issue('—','Unassigned Captain','Mission has no Captain assigned',mid)
        elif c not in pilots or pilots[c].role!='CAPT': issue(c,'Crew qualification','Captain role missing or incorrect',mid)
        if not f: issue('—','Unassigned First Officer','Mission has no First Officer assigned',mid)
        elif f not in pilots or pilots[f].role!='FO': issue(f,'Crew qualification','First Officer role missing or incorrect',mid)
        for pid in (c,f):
            if pid in pilots and not opt.available(pid,m,avail):
                issue(pid,'Pilot availability','Day off, leave or unavailable on mission date',mid)
        if c in pilots and f in pilots and not (pilots[c].experienced or pilots[f].experienced):
            issue(f'{c} / {f}','Crew experience','At least one experienced pilot required',mid)
    if not missions:
        issue('—','Data','No active missions found')
        return {'issues':issues,'monthly_not_assessed':[]}
    start=min(m.departure for m in missions)
    end=max(m.arrival for m in missions)
    monthly_not_assessed=[]
    for pid,pilot in pilots.items():
        roster=sorted((idx[mid] for mid,crew in mapping.items() if mid in idx and pid in (crew['captain'],crew['fo'])),key=lambda m:m.departure)
        for earlier,later in zip(roster,roster[1:]):
            if opt.overlap(opt.duty_interval(earlier),opt.duty_interval(later)):
                issue(pid,'Overlapping duties',f'{earlier.id} / {later.id}',later.id)
        try:
            audit=opt.pilot_ftl_checklist(roster,pilot,avail,start,end)
            for rule,ok in audit['checks'].items():
                if not ok: issue(pid,rule,', '.join(m.id for m in roster) or 'No flights')
            status=str(audit['monthly_status']).upper()
            if status.startswith('FAIL') or status=='NIET OK':
                issue(pid,'Two 48-hour rest periods per complete calendar month','Monthly recovery rest requirement')
            elif 'NOT ASSESSED' in status or 'NIET BEOORDEELD' in status:
                monthly_not_assessed.append(pid)
        except Exception as exc:
            issue(pid,'FTL calculation error',str(exc))
    return {'issues':issues,'monthly_not_assessed':monthly_not_assessed,
            'mission_count':len(idx),'assignment_count':len(mapping)}

def pilot_schedule_rows(solution, missions, pilots):
    """Derive the scratch pilot roster from saved mission crew assignments only."""
    mission_by_id={m.id:m for m in missions}
    grouped={pid:[] for pid in pilots}
    for mid,crew in assignments(solution).items():
        if mid not in mission_by_id:
            continue
        m=mission_by_id[mid]
        for role,pid in (('CAPT',crew['captain']),('FO',crew['fo'])):
            if pid in grouped:
                grouped[pid].append((m,role,crew['aircraft']))
    rows=[]
    for pid,p in sorted(pilots.items()):
        flights=sorted(grouped[pid],key=lambda x:x[0].departure)
        if not flights:
            rows.append({'Pilot':pid,'Role':p.role,'Experience':'Experienced' if p.experienced else 'Inexperienced',
                         'Mission':'—','Aircraft':'—','Departure':'—','Arrival':'—','Route':'—'})
        for m,role,reg in flights:
            rows.append({'Pilot':pid,'Role':role,'Experience':'Experienced' if p.experienced else 'Inexperienced',
                         'Mission':m.id,'Aircraft':reg,'Departure':m.departure.isoformat(sep=' ',timespec='minutes'),
                         'Arrival':m.arrival.isoformat(sep=' ',timespec='minutes'),
                         'Route':f'{m.origin} → {m.destination}'})
    return rows


def render_scratch_data(solution, missions, pilots, board_name):
    """All views use the identical, currently saved scratch solution."""
    st.markdown('### Scratch board data')
    mission_by_id={m.id:m for m in missions}
    crew=assignments(solution)
    mission_rows=[]
    for mid,entry in sorted(crew.items(),key=lambda item:mission_by_id[item[0]].departure if item[0] in mission_by_id else datetime.max):
        m=mission_by_id.get(mid)
        mission_rows.append({'Mission':mid,'Aircraft':entry['aircraft'],
                             'Departure':m.departure.isoformat(sep=' ',timespec='minutes') if m else '—',
                             'Arrival':m.arrival.isoformat(sep=' ',timespec='minutes') if m else '—',
                             'Captain':entry['captain'] or 'UNASSIGNED',
                             'First Officer':entry['fo'] or 'UNASSIGNED'})
    tab_m,tab_p=st.tabs(['Mission assignments','Pilot planning'])
    with tab_m:
        st.dataframe(pd.DataFrame(mission_rows),hide_index=True,use_container_width=True)
    with tab_p:
        rows=pilot_schedule_rows(solution,missions,pilots)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('These pilot assignments belong only to this scratch board; original pilot data is unchanged.')


def render(source, pilot_label, swaps_fn, swaps_render):
    st.markdown('#### Scratch boards')
    st.caption('Independent crew scenarios · original optimizer results are read-only. Flight times and aircraft stay fixed.')
    board_names=names()
    with st.expander('Create scratch board',expanded=not bool(board_names)):
        proposed=st.text_input('Board name',value='Scratch_1',key='scratch_create_name')
        if st.button('Create from displayed optimizer solution',key='scratch_create'):
            try:
                create(proposed,source)
                st.rerun()
            except Exception as exc: st.error(str(exc))
    if not board_names: return
    chosen=st.selectbox('Active scratch board',board_names,key='scratch_selected')
    data=read(chosen)
    sol=data['solution']
    a=assignments(sol)
    st.caption(f'{len(a)} assigned missions · {len(swaps_fn(sol))} outstation swap events · Last saved: {data.get("updated_at", "—")}')
    mids=sorted(a)
    if mids:
        mid=st.selectbox('Select mission to change crew',mids,key='scratch_mission')
        import optimizer as opt
        _,pilots,_=opt.load_data()
        captains=sorted(pid for pid,p in pilots.items() if p.role=='CAPT')
        fos=sorted(pid for pid,p in pilots.items() if p.role=='FO')
        if captains and fos:
            c1,c2=st.columns(2)
            cur=a[mid]
            c=c1.selectbox('Captain',captains,index=captains.index(cur['captain']) if cur['captain'] in captains else 0,
                            format_func=lambda x:pilot_label(x),key=f'scratch_c_{chosen}_{mid}')
            f=c2.selectbox('First Officer',fos,index=fos.index(cur['fo']) if cur['fo'] in fos else 0,
                            format_func=lambda x:pilot_label(x),key=f'scratch_f_{chosen}_{mid}')
            b1,b2,b3=st.columns(3)
            if b1.button('Apply crew change',key='scratch_apply'):
                if c==cur['captain'] and f==cur['fo']:
                    st.info('No change to apply.')
                else:
                    new=edit(sol,mid,c,f)
                    history=data.get('history',[])
                    history.append(sol)
                    write(chosen,new,history[-20:],None)
                    st.rerun()
            if b2.button('Undo last change',disabled=not bool(data.get('history')),key='scratch_undo'):
                history=data['history']; prior=history.pop()
                write(chosen,prior,history,None);st.rerun()
            if b3.button('Validate board',key='scratch_validate'):
                result=validate(sol);write(chosen,sol,data.get('history',[]),result);st.rerun()
    result=data.get('validation')
    if result is None: st.warning('NOT VALIDATED — validate after any edit. Scratch boards are not operational schedules.')
    elif result['issues']:
        st.error(f'INVALID — {len(result["issues"])} validation finding(s). Do not use operationally.')
        st.dataframe(pd.DataFrame(result['issues']),hide_index=True,use_container_width=True)
    else:
        st.success('All assessed hard crew and FTL checks passed.')
    if result and result.get('monthly_not_assessed'):
        st.caption('Monthly two-48-hour rest rule: NOT ASSESSED for partial calendar months.')
    st.caption('Manual crew edits do not model pilot positioning/deadhead travel. This validator does not certify that a pilot can physically reach the next aircraft; review outstation handovers separately.')
    with st.expander('Scratch board outstation swaps',expanded=False):
        swaps_render(sol)
    if st.button('Delete this scratch board',key='scratch_delete'):
        delete(chosen);st.rerun()
    return sol

# v2: dedicated browser-tab workspace, using the original operations board renderer.
def render_launcher(source):
    import html
    from urllib.parse import quote
    st.markdown('### Scratch boards')
    st.caption('Each scenario opens in a separate browser tab. Aircraft, flights and times remain fixed.')
    existing = names()
    with st.form('scratch_new_board_v2', clear_on_submit=False):
        proposed = st.text_input('New scenario name', value='Scratch_1')
        submitted = st.form_submit_button('Create scratch board')
    if submitted:
        try:
            create(proposed, source)
            st.success('Scenario created. Open it below in a new tab.')
            existing=names()
        except Exception as exc:
            st.error(str(exc))
    if existing:
        selected=st.selectbox('Saved scratch boards',existing,key='scratch_open_board_v2')
        url='?scratch_board='+quote(selected,safe='')
        st.markdown(f'<a href="{html.escape(url,quote=True)}" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:0.55rem 1rem;border-radius:0.5rem;background:#1764ae;color:white;text-decoration:none;font-weight:600">Open {html.escape(selected)} in new tab ↗</a>',unsafe_allow_html=True)


def render_workspace(name, board_renderer, missions_loader, pilot_label, swaps_fn, swaps_render):
    import html
    from urllib.parse import quote
    safe=_safe(name)
    if safe not in names():
        st.error('Scratch board not found. Return to the main Mission Planning Board.')
        return
    data=read(safe)
    sol=data['solution']
    st.title(f'Scratch Board · {safe}')
    st.caption('Independent crew scenario · flights, aircraft and times are fixed · click a mission block on the timeline to edit its crew')
    names_all=names()
    top=st.columns([3,1,1])
    if len(names_all)>1:
        other=top[0].selectbox('Switch scratch board',names_all,index=names_all.index(safe))
        if other!=safe:
            st.query_params['scratch_board']=other
            st.rerun()
    else:
        top[0].caption('Open additional scenarios from the main Mission Planning Board.')
    if top[1].button('Duplicate board',use_container_width=True):
        candidate=safe+'_copy'
        n=2
        while candidate in names():
            candidate=safe+f'_copy{n}';n+=1
        create(candidate,sol)
        st.query_params['scratch_board']=candidate
        st.rerun()
    if top[2].button('Check FTL',type='primary',use_container_width=True):
        with st.spinner('Checking all pilots and mission assignments...'):
            result=validate(sol)
        write(safe,sol,data.get('history',[]),result)
        st.rerun()
    result=data.get('validation')
    if result is None:
        st.warning('NOT VALIDATED — run Check FTL after any crew change.')
    elif result.get('issues'):
        st.error(f'{len(result["issues"])} finding(s) — this scenario is NOT valid.')
    else:
        st.success('No violations found by the implemented crew/FTL checks. Pilot positioning is not assessed.')
    assignment=assignments(sol)
    import optimizer as opt
    try:
        missions_all,pilots,availability=opt.load_data()
    except Exception as exc:
        st.error(f'Cannot load pilot data: {exc}')
        return

    @st.dialog('Edit mission crew', width='large')
    def crew_dialog(mission):
        # Reload on dialog open to avoid overwriting changes from another tab.
        latest=read(safe)
        latest_sol=latest['solution']
        current=assignments(latest_sol).get(mission)
        if current is None:
            st.error('Mission no longer exists in this scratch board.')
            return
        st.markdown(f'**Mission {mission}** · Aircraft {current["aircraft"]}')
        eligible,blocked=eligible_crew(latest_sol,mission,missions_all,pilots,availability)
        caps=eligible['CAPT']; fos=eligible['FO']
        # Empty assignment is always allowed in a scratch scenario.
        # The FTL validator flags any unfilled seat as a hard finding.
        st.caption(f'Available crew: {len(caps)} Captains / {len(fos)} First Officers. Day off, leave and overlapping duties excluded.')
        if (current['captain'] and current['captain'] not in caps) or (current['fo'] and current['fo'] not in fos):
            st.warning('The current crew contains an unavailable pilot. Select a valid replacement.')
        with st.form(f'scratch_popup_form_{safe}_{mission}'):
            col1,col2=st.columns(2)
            captain_options=['']+caps
            fo_options=['']+fos
            label_or_empty=lambda pid: '— UNASSIGNED / Release pilot —' if not pid else pilot_label(pid)
            c=col1.selectbox('Captain',captain_options,index=captain_options.index(current['captain']) if current['captain'] in captain_options else 0,format_func=label_or_empty)
            f=col2.selectbox('First Officer',fo_options,index=fo_options.index(current['fo']) if current['fo'] in fo_options else 0,format_func=label_or_empty)
            st.caption('Choose UNASSIGNED to release a crew member. An incomplete mission is saved as a draft and fails Check FTL until both seats are filled.')
            submit=st.form_submit_button('Save crew assignment',type='primary',use_container_width=True)
        if submit:
            try:
                verify_crew_change(latest_sol,mission,c,f,missions_all,pilots,availability)
                if (c,f)!=(current['captain'],current['fo']):
                    updated=edit(latest_sol,mission,c,f)
                    history=latest.get('history',[])+[latest_sol]
                    write(safe,updated,history[-40:],None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        with st.expander('Why are other pilots not listed?'):
            st.caption('Only available, role-qualified pilots can be assigned. UNASSIGNED releases a pilot for other missions. A mission missing CAPT or FO fails validation. Experience is checked once both seats are filled.')

    selected_event=None
    try:
        missions=missions_loader()
        selected_event=board_renderer(sol,missions,'SCRATCH BOARD · '+safe,scratch_board_name=safe)
    except Exception as exc:
        st.error(f'Cannot render planning board: {exc}')
    if isinstance(selected_event,dict):
        mid=str(selected_event.get('mission',''))
        nonce=selected_event.get('nonce')
        event_key=f'_scratch_handled_event_{safe}'
        if mid in assignment and nonce!=st.session_state.get(event_key):
            st.session_state[event_key]=nonce
            crew_dialog(mid)

    render_scratch_data(sol, missions_all, pilots, safe)

    with st.expander('Alternative: select mission manually', expanded=False):
        mids=sorted(assignment)
        if mids:
            manual=st.selectbox('Mission',mids,format_func=lambda m:f'{m} · {assignment[m]["aircraft"]}',key=f'manual_scratch_{safe}')
            if st.button('Edit selected mission',key=f'manual_edit_{safe}'):
                crew_dialog(manual)

    a,b=st.columns(2)
    if a.button('Undo last change',disabled=not bool(data.get('history')),use_container_width=True):
        hist=data['history'];prior=hist.pop();write(safe,prior,hist,None);st.rerun()
    with b.popover('Board actions',use_container_width=True):
        new_name=st.text_input('Rename board',value=safe,key='scratch_rename_input')
        if st.button('Rename',key='scratch_rename'):
            new_safe=_safe(new_name)
            if new_safe!=safe and new_safe in names():st.error('Name already exists.')
            elif new_safe!=safe:
                write(new_safe,sol,data.get('history',[]),result);delete(safe)
                st.query_params['scratch_board']=new_safe;st.rerun()
        if st.button('Delete board',key='scratch_delete_workspace'):
            delete(safe);st.query_params.clear();st.rerun()
    if result is not None:
        st.markdown('### FTL validation findings')
        if result.get('issues'):
            st.dataframe(pd.DataFrame(result['issues']),hide_index=True,use_container_width=True)
        else:
            st.info('No issues reported by the implemented checks.')
        if result.get('monthly_not_assessed'):
            st.warning('Monthly 48-hour recovery rule NOT ASSESSED for partial calendar months: '+', '.join(result['monthly_not_assessed']))
    st.caption('Important: deadhead/positioning between aircraft is not checked. A passing FTL check is not operational certification.')
    with st.expander('Outstation crew swaps'):
        swaps_render(sol)
