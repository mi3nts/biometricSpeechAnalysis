# Random Forest

This was initially built in Ggoogle Colab and was developed in a notebook style format.
It has been modified to run as a single script, but to tinker with it I would recommend
looking at the link at the bottom of the page

## Instructions
Install the python packages required with

`pip install -r requirements.txt` 

Also get the spacy data

`python -m spacy download en_core_web_sm`

Extract the MINTS.zip archive of the data such that the directory structure contains a file in the current working directory called
`nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr`


Run the program with
`python randomforesttoxicity.py`

The model is loaded from a pickle so you do not have to train it each time.


## Results
Some graphs will be shown, they're included in the images subdirectory

What the model learned as the importance of each feature, along with a short description:

```
********** Importances of each feature **********
('FT8', 0.11398160601212044) in the Brocas Area (ba47R) on the Brocas Area which is Responsible for speech production
('P5', 0.10907992368326518) in the Somatic Sensory Association Area (ba39L) on the Wernickes Area which is Involved in understanding speech
('FT9', 0.08896841332454014) in the Auditory Association Area (ba20L) on the Temporal which is Brain Activity
('P3', 0.07972376362595752) in the Somatic Sensory Association Area (ba39L) on the Wernickes Area which is Involved in understanding speech
('FC4', 0.07462683507821186) in the Primary Motor Cortex (ba06R) on the Premotor Cortex which is Involved in planning of movement
('Oz', 0.07141589483445351) in the Visual Cortex (ba17R) on the Visual Cortex which is Processes visual stimuli
('C4', 0.06815517510399283) in the Primary Sensory Cortex (ba01R) on the Primary Sensory Cortex which is Main receptive area for the senses, especially touch
('P1', 0.06771802972547668) in the Somatic Sensory Association Area (ba07L) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('F1', 0.0461280038536156) in the Premotor Cortex (ba08L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('POz', 0.03760837127206198) in the Visual Association Area (ba17L) on the Visual Cortex which is Processes visual stimuli
('CP5', 0.020466400133619965) in the Somatic Sensory Association Area (ba40L) on the Wernickes Area which is Involved in understanding speech
('FCz', 0.014836137707942025) in the Primary Motor Cortex (ba06R) on the Premotor Cortex which is Involved in planning of movement
('Cz', 0.014095983647985642) in the Primary Sensory Cortex (ba05L) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('Pz', 0.011737474043912306) in the Somatic Sensory Association Area (ba07R) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('C5', 0.011533206931747163) in the Primary Sensory Cortex (ba42L) on the Auditory Cortex which is Processes sound
('C2', 0.010648327340658539) in the Primary Sensory Cortex (ba05R) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('P6', 0.00896083917601965) in the Somatic Sensory Association Area (ba39R) on the Wernickes Area which is Involved in understanding speech
('CP3', 0.008528987717932588) in the Somatic Sensory Association Area (ba02L) on the Primary Sensory Cortex which is Main receptive area for the senses, especially touch
('F4', 0.007718206095560837) in the Premotor Cortex (ba08R) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('FC3', 0.007212499471724588) in the Primary Motor Cortex (ba06L) on the Premotor Cortex which is Involved in planning of movement
('AF4', 0.007115007743437316) in the Prefrontal Cortex (ba09R) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('FC5', 0.006749745644479902) in the Primary Motor Cortex (BROCLA) on the Brocas Area which is Responsible for speech production
('FC6', 0.006522393910989322) in the Primary Motor Cortex (ba44R) on the Frontal Cortex which is Brain Activity
('PO7', 0.006056026705019601) in the Visual Association Area (ba19L) on the Visual Association Area which is Involved in high level processing of visual stimuli
('F7', 0.0057365015249150365) in the Premotor Cortex (ba47L) on the Brocas Area which is Responsible for speech production
('Fz', 0.005068354670064853) in the Premotor Cortex (ba08L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('FT10', 0.004930592553603733) in the Auditory Association Area (ba20R) on the Temporal which is Brain Activity
('P4', 0.004795931422598801) in the Somatic Sensory Association Area (ba39R) on the Wernickes Area which is Involved in understanding speech
('O2', 0.004048163188204925) in the Visual Cortex (ba18R) on the Visual Cortex which is Processes visual stimuli
('F8', 0.004015900953070555) in the Premotor Cortex (ba45R) on the Frontal Cortex which is Brain Activity
('Fp1', 0.003872522874342877) in the Prefrontal Cortex (ba10L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('TP8', 0.0035707167771813946) in the Wernickes Area (ba21R) on the Temporal which is Brain Activity
('AFz', 0.0035317152282445673) in the Prefrontal Cortex (ba09L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('Fp2', 0.0034514858706774904) in the Prefrontal Cortex (ba10R) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('PO8', 0.003185019809554591) in the Visual Association Area (ba19R) on the Visual Association Area which is Involved in high level processing of visual stimuli
('C3', 0.0027973587546788687) in the Primary Sensory Cortex (ba02L) on the Primary Sensory Cortex which is Main receptive area for the senses, especially touch
('AF7', 0.0027708625852257227) in the Prefrontal Cortex (ba46L) on the Frontal Cortex which is Brain Activity
('C1', 0.0027609476668694383) in the Primary Sensory Cortex (ba05L) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('F6', 0.0027571355890615903) in the Premotor Cortex (ba46R) on the Frontal Cortex which is Brain Activity
('AF8', 0.0026781595699734997) in the Prefrontal Cortex (ba46R) on the Frontal Cortex which is Brain Activity
('P7', 0.002648114600779758) in the Somatic Sensory Association Area (ba37L) on the Temporal which is Brain Activity
('C6', 0.0025635653443479844) in the Primary Sensory Cortex (ba41R) on the Auditory Cortex which is Processes sound
('PO4', 0.002226614750227205) in the Visual Association Area (ba19R) on the Visual Association Area which is Involved in high level processing of visual stimuli
('TP7', 0.0021440263617554774) in the Wernickes Area (ba21L) on the Temporal which is Brain Activity
('CP4', 0.0020490685156979147) in the Somatic Sensory Association Area (ba40R) on the Wernickes Area which is Involved in understanding speech
('CP2', 0.0019633403800474384) in the Somatic Sensory Association Area (ba05R) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('F5', 0.001888924748497049) in the Premotor Cortex (ba46L) on the Frontal Cortex which is Brain Activity
('PO3', 0.0018253253683519822) in the Visual Association Area (ba19L) on the Visual Association Area which is Involved in high level processing of visual stimuli
('P8', 0.001803581324583354) in the Somatic Sensory Association Area (ba37R) on the Temporal which is Brain Activity
('CP6', 0.0017830242512797835) in the Somatic Sensory Association Area (ba40R) on the Wernickes Area which is Involved in understanding speech
('CPz', 0.0017816353327192375) in the Somatic Sensory Association Area (ba05R) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('AF3', 0.001739471671038268) in the Prefrontal Cortex (ba09L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('F2', 0.0017041107557588641) in the Premotor Cortex (ba08R) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('O1', 0.0016028300706485985) in the Visual Cortex (ba19L) on the Visual Association Area which is Involved in high level processing of visual stimuli
('FC2', 0.0015938850733130166) in the Primary Motor Cortex (ba06R) on the Premotor Cortex which is Involved in planning of movement
('T8', 0.0015888076035567111) in the Auditory Cortex (ba21R) on the Temporal which is Brain Activity
('Fpz', 0.0015021004910941774) in the Prefrontal Cortex (ba10L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('TP10', 0.0014650119142988463) in the Wernickes Area (ba21R) on the Temporal which is Brain Activity
('FT7', 0.0014118782484612183) in the Brocas Area (ba47L) on the Brocas Area which is Responsible for speech production
('P2', 0.0013916145214560368) in the Somatic Sensory Association Area (ba07R) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('F3', 0.0012019655344660128) in the Premotor Cortex (ba08L) on the Prefrontal Cortex which is Involved in decision making and abstract thought
('FC1', 0.0011788980070306107) in the Primary Motor Cortex (ba06L) on the Premotor Cortex which is Involved in planning of movement
('CP1', 0.0009420203798532115) in the Somatic Sensory Association Area (ba05L) on the Somatic Sensory Association Area which is Involved in high level touch interpretation
('HR', 0.00018312899593152048)
('Resp.', 0.00017961579772685427)
('Temp.', 4.062168634464409e-05)
('GSR', 1.9301462928879072e-05)
('ECG.', 1.8810057581009738e-05)
('PPG', 4.9577737538347884e-08)
('SpO2', 3.534949363910001e-08)
('ExG 2', 1.2756587890589767e-20)
('ExG 4', 9.736376515872627e-21)
('ExG 3', 4.8450580632276565e-21)
('ExG 1', -2.8557786538928662e-21)
```


## Link to Online Environment
To run this visualization tool without installing it on your local machine, please use
[this link](https://colab.research.google.com/drive/1BQVIDoun1ZvuyVa916Qr_CayRXto6_9Y#scrollTo=PSVr82bmFaHV) to access the hosted
version on Google Colab.
