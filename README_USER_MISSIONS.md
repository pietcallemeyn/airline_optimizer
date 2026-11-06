# Crew Planning v3.2 — persoonlijke missie-input

## Wijziging

Na aanmelden krijgt iedere gebruiker een eigen `output/users/<storage-key>/input/`:

- `missions.csv` — de actuele, bewerkbare missies van die gebruiker.
- `missions_source.json` — de naam van de gebruikte importbron.

**Het originele XLS/XLSX-bestand wordt niet bewaard.** De app converteert het in het geheugen en slaat alleen de CSV en bronvermelding op.

De eerste keer dat een gebruiker inlogt, wordt de huidige gedeelde `data/missions.csv` eenmalig gekopieerd naar zijn persoonlijke inputmap (of `data/missions_default.csv` als die ontbreekt). Vanaf dat moment blijven wijzigingen van andere gebruikers gescheiden. 'Restore standard missions' herstelt voor de actieve gebruiker vanuit `data/missions_default.csv`.

De optimizer ontvangt via `CREW_MISSIONS_FILE` het persoonlijke CSV-pad. De overige referentiedata (pilots, aircraft, airports, pilot availability) blijven gedeeld. Een losse terminal-run van `optimizer.py` zonder die variabele gebruikt zoals vroeger `data/missions.csv`.

## Installatie

1. Maak een back-up van de bestaande app en `output/`.
2. Vervang `app.py` en `optimizer.py` uit deze ZIP. De overige bestanden zijn meegeleverd voor compatibiliteit.
3. Behoud `auth_data/users.json`, `output/users/`, en je bestaande `data/`.
4. Herstart Streamlit en meld je aan. Elke gebruiker krijgt automatisch zijn eigen inputmap.
5. Start de optimizer opnieuw na het wijzigen van je missies.

## Let op

- Bestaande gebruikers krijgen bij de eerste start met v3.2 een kopie van de *huidige gedeelde* `data/missions.csv`. Controleer of dit de gewenste uitgangsset is. Als verschillende gebruikers vroeger andere missiegegevens verwachtten, zet dan vooraf de juiste CSV in hun eigen `input/missions.csv`.
- De gedeelde pilootbeschikbaarheid en overige referentiedata zijn **niet** per gebruiker gescheiden.
- Reeds bestaande optimizerresultaten worden niet automatisch aan een specifieke missieversie gekoppeld; na een import opnieuw optimaliseren.
- De oorspronkelijke XLSX wordt bewust niet bewaard; `missions_source.json` bewaart alleen de bestandsnaam, niet de inhoud.
