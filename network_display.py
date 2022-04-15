import json

import dash
import dash_cytoscape as cyto
# import dash_html_components as html
# import dash_core_components as dcc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import itertools
import pandas as pd

df = pd.read_csv("assets\EEG01_chanlocs_cartesian.txt", delim_whitespace=True, names=['Name', 'x', 'y', 'z'])

cyto.load_extra_layouts()

brainmap = {
    'Fpz':'Prefrontal Cortex',
    'Fp1':'Prefrontal Cortex',
    'Fp2':'Prefrontal Cortex',
    'AF7':'Prefrontal Cortex',
    'AF3':'Prefrontal Cortex',
    'AFz':'Prefrontal Cortex',
    'AF4':'Prefrontal Cortex',
    'AF8':'Prefrontal Cortex',
    'F7' :'Premotor Cortex',
    'F5':'Premotor Cortex',
    'F3':'Premotor Cortex',
    'F1':'Premotor Cortex',
    'Fz':'Premotor Cortex',
    'F2':'Premotor Cortex',
    'F4':'Premotor Cortex',
    'F6':'Premotor Cortex',
    'F8':'Premotor Cortex',
    'FT9':'Auditory Association Area',
    'FT7':'Brocas Area',
    'FC5':'Primary Motor Cortex',
    'FC3':'Primary Motor Cortex',
    'FC1':'Primary Motor Cortex',
    'FCz':'Primary Motor Cortex',
    'FC2':'Primary Motor Cortex',
    'FC4':'Primary Motor Cortex',
    'FC6':'Primary Motor Cortex',
    'FT8': 'Brocas Area',
    'FT10':'Auditory Association Area',
    'T7':'Auditory Cortex',
    'C5':'Primary Sensory Cortex',
    'C3':'Primary Sensory Cortex',
    'C1':'Primary Sensory Cortex',
    'Cz':'Primary Sensory Cortex',
    'C2':'Primary Sensory Cortex',
    'C4':'Primary Sensory Cortex',
    'C6':'Primary Sensory Cortex',
    'T8':'Auditory Cortex',
    'TP7':'Wernickes Area',
    'CP5':'Somatic Sensory Association Area',
    'CP3':'Somatic Sensory Association Area',
    'CP1':'Somatic Sensory Association Area',
    'CPz':'Somatic Sensory Association Area',
    'CP2':'Somatic Sensory Association Area',
    'CP4':'Somatic Sensory Association Area',
    'CP6':'Somatic Sensory Association Area',
    'TP8':'Wernickes Area',
    'TP10':'Wernickes Area',
    'P7':'Somatic Sensory Association Area',
    'P5':'Somatic Sensory Association Area',
    'P3':'Somatic Sensory Association Area',
    'P1':'Somatic Sensory Association Area',
    'Pz':'Somatic Sensory Association Area',
    'P2':'Somatic Sensory Association Area',
    'P4':'Somatic Sensory Association Area',
    'P6':'Somatic Sensory Association Area',
    'P8':'Somatic Sensory Association Area',
    'PO7':'Visual Association Area',
    'PO3':'Visual Association Area',
    'POz':'Visual Association Area',
    'PO4':'Visual Association Area',
    'PO8':'Visual Association Area',
    'O1':'Visual Cortex',
    'Oz':'Visual Cortex',
    'O2':'Visual Cortex',
    'REF':'NAN',
    'GND':'NAN'
    }
broadmannmapping ={
    'Fpz':'ba10L',
    'Fp1':'ba10L',
    'Fp2':'ba10R',
    'AF7':'ba46L',
    'AF3':'ba09L',
    'AFz':'ba09L',
    'AF4':'ba09R',
    'AF8':'ba46R',
    'F7' :'ba47L',
    'F5':'ba46L',
    'F3':'ba08L',
    'F1':'ba08L',
    'Fz':'baO8L',
    'F2':'ba08R',
    'F4':'ba08R',
    'F6':'ba46R',
    'F8':'ba45R',
    'FT9':'ba20L',
    'FT7':'ba47L',
    'FC5':'BROCLA',
    'FC3':'ba06L',
    'FC1':'ba06L',
    'FCz':'ba06R',
    'FC2':'ba06R',
    'FC4':'ba06R',
    'FC6':'ba44R',
    'FT8': 'ba47R',
    'FT10':'ba20R',
    'T7':'ba42L',
    'C5':'ba42L',
    'C3':'ba02L',
    'C1':'ba05L',
    'Cz':'ba05L',
    'C2':'ba05R',
    'C4':'ba01R',
    'C6':'ba41R',
    'T8':'ba21R',
    'TP7':'ba21L',
    'CP5':'ba40L',
    'CP3':'ba02L',
    'CP1':'ba05L',
    'CPz':'ba05R',
    'CP2':'ba05R',
    'CP4':'ba40R',
    'CP6':'ba40R',
    'TP8':'ba21R',
    'TP10':'ba21R',
    'P7':'ba37L',
    'P5':'ba39L',
    'P3':'ba39L',
    'P1':'ba07L',
    'Pz':'ba07R',
    'P2':'ba07R',
    'P4':'ba39R',
    'P6':'ba39R',
    'P8':'ba37R',
    'PO7':'ba19L',
    'PO3':'ba19L',
    'POz':'ba17L',
    'PO4':'ba19R',
    'PO8':'ba19R',
    'O1':'ba19L',
    'Oz':'ba17R',
    'O2':'ba18R',
    'REF':'NAN',
    'GND':'NAN'
    }
broadmanntoarea ={
    'ba01':'Primary Sensory Cortex',
    'ba02':'Primary Sensory Cortex',
    'ba03':'Primary Sensory Cortex',
    'ba04':'Primary Motor Cortex',
    'ba05':'Somatic Sensory Association Area',
    'ba06':'Premotor Cortex',
    'ba07':'Somatic Sensory Association Area',
    'ba08':'Prefrontal Cortex',
    'ba09':'Prefrontal Cortex',
    'ba10':'Prefrontal Cortex',
    'ba11':'Prefrontal Cortex',
    'ba12':'Prefrontal Cortex',
    'ba17':'Visual Cortex',
    'ba18':'Visual Cortex',
    'ba19':'Visual Association Area',
    'ba20':'Temporal',
    'ba21':'Temporal',
    'ba22':'Wernickes Area',
    'ba37':'Temporal',
    'ba38':'Temporal',
    'ba39':'Wernickes Area',
    'ba40':'Wernickes Area',
    'ba41':'Auditory Cortex',
    'ba42':'Auditory Cortex',
    'ba43':'Frontal Cortex',
    'ba44':'Frontal Cortex',
    'BROCLA':'Brocas Area',
    'ba45':'Frontal Cortex',
    'ba46':'Frontal Cortex',
    'ba47':'Frontal Cortex',
    }
areamap = {
    'Prefrontal Cortex':'Involved in decision making and abstract thought',
    'Premotor Cortex':'Involved in planning of movement',
    'Brocas Area':'Responsible for speech production',
    'Auditory Cortex':'Processes sound',
    'Auditory Association Area':'Responsible for high level processing of sound, such as memory',
    'Primary Motor Cortex':'Executes Movement',
    'Primary Sensory Cortex':'Main receptive area for the senses, especially touch',
    'Wernickes Area':'Involved in understanding speech',
    'Somatic Sensory Association Area':'Involved in high level touch interpretation',
    'Visual Association Area':'Involved in high level processing of visual stimuli',
    'Visual Cortex':'Processes visual stimuli'
    }

# build nodes and edges, including the positions the electrodes
# this function takes in full_list: a list of lists that contains electrodes that were the three most active for each epoch. e.g.: [[Fp1, Fp2, Fz],[AF8, TP7, PO8]]
def nodesEdges(full_list):


    # creating nodes i.e. electrodes with their respective positions as described by channel location text file
    nodes = [
        {
            'data': {'id': short, 'label': short},
            'position': {'x': x, 'y': y}
        }
        for short, short, y, x in (
            list(zip(df["Name"],df["Name"],-df["x"],-df["y"]))
        )
    ]

    edges_all = []
    
    # creating the pairs of edges between electrodes in each epoch in (source, target) format
    for epoch in full_list:
        edges_all.extend(list(itertools.combinations(epoch,2)))


    edges = [
        {'data': {'source': source, 'target': target}}
        for source, target in (
            edges_all
        )
    ]

    return nodes, edges

def genStylesheet():
    node_text_color = 'blue'
    node_color = 'blue'
    node_size = '15px'
    node_text_size = '5px'
    node_border_color_alpha = 'blue'
    
    # initially all electrodes will be grey
    stylesheet = [
        {'selector': 'node',
            'style': 
            {'label': 'data(label)',
                'width': node_size,
                'height': node_size,
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "5px",
                'shape':'circle',
                'background-color': 'gray'}
        },
    ]

    return stylesheet


electrodes = ['Fp1', 'Fp2', 'Fz']

nodes = nodesEdges(electrodes)[0]
stylesheet = genStylesheet()

# highlight the current nodes by updating their stylesheets
for i in electrodes:
    stylesheet.append({
        'selector': f'[label = "{i}"]',
        'style':
            {
                'background-color': 'red',
                'z-index': 9999
            }
    })
app = dash.Dash(__name__)
elems = nodes 
        
app.layout = html.Div(children=[
     # Creating the default networks with default stylesheet. Only the relevant (epoch-specific) edges and nodes will get highlighted due to callbacks
     cyto.Cytoscape(
         id='network',
         layout={'name': 'preset'},
         elements=elems,
         stylesheet=stylesheet,
         style={
             'width': '450px', 
             'height': '450px',
             'display': 'inline-block'}
    )
])


app.run_server(debug=True)