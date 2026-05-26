# Bakalaurinis-darbas

## Duomenų sukėlimas į vieną failą

Pradiniame duomenų rinkinyje turėjome daugybę .mat failų aprašytų pagal kiekvienam stebimam signalui pritaikytus parametrus: gedimo tipas, sūkiai, apkrova, bandymo skaičius. Tad buvo nuspręsta šiuos visus duomenis apjungti ir sukelti į vieną lentelę .parquet formatu. Programinį kodą galima pamatyti `parquet_failo_konstravimas.ipynb` faile.

## Pirminė duomenų analizė

Po failo sukonstravimo toliau galima naudotis `pirmine_duomenu_analize`, kad atlikti pirminę duomenų analizę pavaizduotą rašto darbe.

## Duomenų transformavimas MultiRocket ir GWAN metodams

Kadangi šie metodai naudoja (MultiRocket .ts ir GWAN .pkl) skirtingus failų tipus, prieš apmokant modelį juos reikia patalpinti tuose formatuose. Papildomai GWAN atveju, prieš patalpinant duomenis į .pkl failą juos reikėjo transformuoti iš signalų į grafų struktūrą. Tam tikslui buvo naudojamas `signals_constr_GWAN`, kuris susegmentavo sinchronizuotus 2 bandymo signalus, o tuomet su `grafu_konstravimas.ipynb` failu sukonstravo grafus. MultiRocket metodui duomenų perkėlimui iš .parquet į .ts formato failą buvo naudojami du: `signals_reconstruction.py` ir `signals_reconstruction_psd.py` failai priklausomai nuo to kokios reprezentacijos buvo naudojamos metodo apmokymui.

## Duomenų transformavimas VibrMamba metodui

VibrMamba kodas naudoja .npy tipo failus, tad tam buvo sukurti `raw_signals_vibrmamba.ipynb` ir `psd_signals_vibrmamba.ipynb` failai, kurių pagalba iš .parquet failo, duomenys talpinami į .npy tipo failus.

## Klasifikavimas naudojant klasikinius ML modelius su PSD reprezentacijomis

PSD_ML failas skirtas apmokyti keturis klasikinius mašininio mokymosi modelius: LinearSVC, LogisticRegression, RandomForest ir KNN. Faile duomenys nuskaitomi iš .parquet failo, paverčiami PSD reprezentacijomis ir jomis apmokomi modeliai.

##MultiRocket, VibrMamba bei GWAN metodų apmokymas

Norint apmokyti MultiRocket metodą, gautus .ts failus reikia patalpinti `MultiRocket/data/uzd_pav` aplanke ir į konsolę parašyti `python main_mtsc.py -d data/ -p uzd_pav -n 30000 -t 6 -s 1 -v 2`. -n parametras nurodo kiek požymių norima išgauti, -t parametras nurodo kiek gijų norima naudoti, -s parametras nurodo ar išsaugoti rezultatus ar ne (0 = neišsaugoti, 1 = išsaugoti) ir -v parametras nurodo ar norima matyti išvestį (rekomenduojama 2, kad matyti išvestį).

Norint apmokyti GWAN metodą, gautus .pkl failus reikia patalpinti `./GWAN-main/GWAN_master/datasets/` aplanke kartu su .py failu kuris paimdamas apmokymo duomenis juos padalina į treniravimo ir validavimo aibes, kurios naudojamos apmokymui. Tuomet su šia komanda konsolėje yra paleidžiamas modelio apmokymas `python ./GWAN-main/GWAN_master/train_graph.py --model_name GWAN --data_name py_failo_pav --data_dir ./GWAN-main/GWAN_master/datasets/duomenu_failo_pav`.

Norint apmokyti VibrMamba metodą yra naudojamas `Virbmamba_realizacija.ipynb` failas.
