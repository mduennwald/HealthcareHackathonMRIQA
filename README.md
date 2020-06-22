## Healthcare Hackathon MRIQA 2020 - Intelligente Qualitätsanalyse von MRT Aufnahmen

*Update 3: erste Ergebniss online*

*Update 2: Benchmark-Datensatz online*

*Update 1: Challenge-Modalitäten spezifiziert*

*Update 0: Einführende [Folien](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/medien/MRIQA_Mainz-Hackathon_2020.pdf) online*

Willkommen auf der Projektseite zu unserem Hack im Bereich Quantenmedizin &amp; künstliche Intelligenz auf dem [Healthcare Hackathon Mainz 2020](https://www.healthcare-hackathon.info/hhmainz). Auf dieser Seite findet Ihr alle notwendigen Informationen zu unserem Thema sowie Kontaktmöglichkeiten.

#### Vorstellung

Der Hack ist eine gemeinsame Initiative der Forschungsgruppe Medizin und Digitalisierung, dem Nationalen Neuroimaging Netzwerk sowie dem Geschäftsbereich IT und Medizintechnik am Universitätsklinikum Magdeburg. Die konkrete Ansprechpartner sind …

| Einrichtung | Otto-von-Guericke-Universität Magdeburg | Deutsches Zentrum für Neurodegenerative Erkrankungen | Universitätsklinikum Magdeburg |
| --- | --- | --- | --- |
| Bereich | [Forschungsgruppe Medizin und Digitalisierung](https://www.kneu.ovgu.de/MedDigit.html) | [Nationalen Neuroimaging Netzwerk](https://www.dzne.de/forschung/neuroimaging/) | [Geschäftsbereich IT und Medizintechnik](https://www.mrz.ovgu.de/) |
| Leitung | [Steffen Oeltze-Jafra](mailto:steffen.oeltze-jafra@med.ovgu.de)  | Emrah Düzel |  Robert Waschipky |
| Ansprechpartner | [Max Dünnwald](mailto:max.duennwald@med.ovgu.de) | [Laura Dobisch](mailto:laura.dobisch@dzne.de) | [Marko Rak](mailto:marko.rak@med.ovgu.de) |

Da wir uns nicht in persona mit euch treffen können, haben wir für den Hack einen Raum im Videokonferenzsystem Zoom bereitgestellt. 

[https://ovgu.zoom.us/j/99914063041](https://ovgu.zoom.us/j/99914063041)

In diesem privaten Raum, werden euch die Ansprechpartner während des Hacks zur Verfügung stehen. Das Raum-Passwort wird zu Beginn des Hackathons verteilt. Falls Ihr es nicht erhalten habt, dann fragt uns bitte direkt an.

#### Motivation

Sowohl bei der Patientenversorgung als auch bei der Durchführung medizinischer Studien spielen Magnetresonanztomographien (MRTs) eine zunehmend wichtigere Rolle. Im Betrieb eines größeren Klinikums oder im Rahmen einer größeren Studie werden tausende von Aufnahmen pro Jahr durchgeführt. Am Universitätsklinikum Magdeburg allein über 11.000 Untersuchungen im Jahr 2019. Ein Teil dieser Aufnahmen stellt sich wegen Artefakten bei der Bildgebung im Nachhinein als qualitativ unzureichend heraus und muss damit ggf. wiederholt werden. Aufgrund der begrenzten MRT-Zeit und hohen MRT-Kosten ist es das Ziel jedes Klinikums/jeder Studie die Anzahl an qualitative unzureichenden Aufnahmen durch Verbesserung des Prozessablaufes, des Personalschulung, der MRT-Parameter usw. zu senken. Als erster Schritt gilt es allerdings dieses Problem ersteinmal erkennen und quantifizieren zu können und dies möglichst vollautomatisch. Darum geht es uns bei diesem Hack.

![MRIQAComp.png](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/medien/MRIQAComp.png)

#### Zielstellung

Entwerft in einem kleinen Team eine künstliche Intelligenz – basierend auf Convolutional Neural Networks (CNNs) – die verschieden Bildgebungsartefakte in MRT-Aufnahmen möglichst verlässlich erkennen und auch unterscheiden kann. Lasst eure künstliche Intelligenz gegen die der anderen Teams antreten. Es geht um Ruhm und Ehre und die beste Lösung wird prämiert…

#### Zeitplanung

Der Zeitplan des Hacks ist wiefolgt

| Datum | Uhrzeit | Inhalt |
| --- | --- | --- |
| 21.06. | 11:30 – 11:40 | Studioauftritt im GHH (Gutenberg Health Hub) |
| 21.06. | Ab 12:00 | Start der Team Coachings |
| 21.06. | Ab 12:00 | Start des Hacks |
| 21.06. | Ab 16:00 | Live-Schaltung Vorstellung Hack |
| 21.06. | Ab 18:00 | Benchmark-Datensatz online |
| 22.06. | 08:00 | Abgabe der Lösungen |
| 22.06. | 12:00 – 13:00 | Prämierung der Teams |

Der Hack startet offiziell am 21.06. ab 12:00. Alle notwendigen Materialen stehen jedoch schon jetzt online zur Verfügung, sodass Ihr euch einen kleinen Vorsprung verschaffen könnt.

#### Voraussetzung

Grundsätzlich sind natürlich alle Hacker herzlich eingeladen teilzunehmen. Folgendes solltet Ihr jedoch mitbringen um richtig durchstarten zu können.

Expertise

- Erfahrungen in der Verarbeitung von Bilddaten
- Erfahrungen im Umgang mit medizinischen Bildern
- Expertise in Deep Learning und Convolutional Neural Networks
- Expertise in Anaconda, Python und PyTorch

Hardware

- Eine Grafikkarte mit viel Speicher
- Je mehr Hauptspeicher desto besser
- Eine verlässliche Kaffeemaschine …

#### Starterset

_Datensatz_

Als Datenbasis für den Hack dient uns der IXI-MRT-Datensatz und zwar genauer gesagt die dort mit T1 und T2 gekennzeichneten Bilddaten. Wir nutzen nur einen Teil dieser beide Datensätze und zwar die des Hammersmith Hospital, diese tragen ein HH im Namen.

[https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)

Ihr könnt euch die Daten natürlich selbst herunterladen und aussortieren, ihr könnt aber auch einfach eines unserer Skripte nutzen. 

_Skripte_

Der IXI-Datensatz hat insgesamt eine hohe Datenqualität, Aufnahmeartefakte sind praktisch nicht vorhanden. Um dennoch eine artefaktbehaftete Datenbasis erzeugen zu können, haben wir euch ein Skript geschrieben

[mriqa_dataset.py](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/starterset/mriqa_dataset.py)

Dieses Skript stellt Methoden zur Verfügung, welche es erlauben um auf die IXI-Bilddaten zuzugreifen und neue Bilddaten inklusive zufälligen Artefakten zu generieren.

Mit der passenden Datenbasis können wir auch schon fast starten. Was wir noch brauchen ist ein Stück künstliche Intelligenz, welche wir auf das Problem loslassen können. Um euch einen ersten Anhaltspunkt zu geben, haben wir euch ein einfaches CNN vorbereitet.

[networks.py](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/starterset/networks.py)

Wir haben Daten und künstliche Intelligenz, jetzt werfen wir alles in einen Topf und mengen das Ganze stundenlang durch bis hoffentlich irgendwas sinnvolles herausgekommen ist. Ein entsprechendes Skript für dieses Rezept findet ihr hier.

[training.py](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/starterset/training.py)

Eins vorweg, wir haben uns so garkeinen Kopf über die richtige Architektur und Parametrierung oder das Thema Data Augmentation gemacht, hier gibt es also noch deutliches Verbesserungspotential.

_Schnellstarter_

Beim ersten Starten des Trainings werdet Ihr vermutlich feststellen, dass Ihr vielleicht noch nicht alle Abhängigkeiten installiert habt. Wer keine Lust hat die Abhängigkeiten selbst zu installieren, der kann sich Arbeit ersparen in dem er sich seine eigene Anaconda Environment bauen lässt.

[conda_instructions](https://github.com/mduennwald/HealthcareHackathonMRIQA/blob/master/starterset/conda_instructions)

Ihr seid startbereit, meldet euch als Team bei Marko Rak und dann kann es schon losgehen!

#### Prämierung

Es geht um die Wurst. Wer am Ende des Hacks die beste Lösung hat wird mit Ruhm und Ehre überschüttet. Nein, Scherz beiseite, jede Lösung und jeder Lösungsweg bringt uns ein Stück weit voran. Aber dennoch werden wir am Ende eine kleine Auswertung auf unseren eigenen echten Bilddaten machen und die beste Lösung/das beste Team prämieren.

#### Einreichung

Sobald eure Lösung einen Stand erreicht hat, den Ihr einreichen wollt, dann lasst uns doch bitte eure Lösung per Commit zukommen. Wenn es die Zeit zulässt machen wir vielleicht schon einen kleinen Testlauf von eurer Lösung auf unseren Bilddaten. Alle Lösungen möchten wir hier sammeln.

[Lösungen](https://github.com/mduennwald/HealthcareHackathonMRIQA/tree/master/einreichungen)

Euer Skript bzw. eure Skripte sollte möglichst genauso funktionieren, wie die von uns zur Verfügung gestellt Skripte, damit wir geringen Aufwand bei der Anpassung auf unsere Daten haben.

#### Auswertung

Um eure Lösungen in einem ersten Schritt bewerten zu können, möchten wir euch bitten eure Lösung auf folgenden Datensatz anzuwenden.

[Benchmark.zip](https://docshare-extern.med.ovgu.de/nextcloud/s/GRJseCCw7iTCBn5)
Passwort: mriqa2020

Für jedes der 1000 MRT-Bilder erhaltet Ihr ein Ergebnis. Dieses übertragt bitte in folgende Tabelle und lasst uns die Tabelle per [E-Mail](mailto:marko.rak@med.ovgu.de) zukommen. 

[Auswertung.xlsx](https://github.com/mduennwald/HealthcareHackathonMRIQA/tree/master/auswertung.xlsx)

Diese werden wir mit unserer eigenen Tabelle vergleichen und herausfinden wie gut eure Lösung ist. Die besten Lösungen werden wir zudem auf unsere Klinikdaten anwenden.

#### Endergebnis

Wir werden sehen wer das Rennen macht…

Auswertung auf dem IXI-Benchmark-Datensatz.

| Top 3 | Teamname | Personen | Ergebnis [%] | Kommentar |
| --- | --- | --- | --- | --- |
| | classicCNN | Max D. | 93,4 | die Baseline |
| 01 | TeamPE | Philipp E. | 96,3 | Transfer Learning |
| 02 | TeamPS | Philipp S. | 95,3 | Residual Learning |
| 03 | TeamLGJS* | Larissa G. & Jan S. | 59,4 | Fourier Analysis |

*mit klassischen nicht deep-learning Methoden!

Auswertung auf unseren internen Daten ohne weiteres Training.

| Top 3 | Teamname | Personen | Ergebnis [%] | Kommentar |
| --- | --- | --- | --- | --- |
| | classicCNN | Max D.  | 39,0 | die Baseline |
| 01 | TeamPE | Philipp E. | 61,2 | Transfer Learning |
| 02 | TeamPS | Philipp S. | 53,0 | Residual Learning |
| 03 | TeamLGJS* | Larissa G. & Jan S. | | Fourier Analysis |

*mit klassischen nicht deep-learning Methoden!

Auswertung auf unseren internen Daten inkl. weiterem Training.

| Top 3 | Teamname | Personen | Ergebnis [%] | Kommentar |
| --- | --- | --- | --- | --- |
| | classicCNN | Max D.  |  | die Baseline |
| 01 | TeamPE | Philipp E. |  | Transfer Learning |
| 02 | TeamPS | Philipp S. |  | Residual Learning |
| 03 | TeamLGJS* | Larissa G. & Jan S. | N.A. | Fourier Analysis |

*mit klassischen nicht deep-learning Methoden!
