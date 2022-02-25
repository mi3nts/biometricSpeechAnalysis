# CODE TO PLOT EXAMPLE DISPLAY USING PLOTLY DASH
# CODE AUTHORED BY: ARJUN SRIDHAR
# PROJECT: biometricSpeechAnalysis
# GitHub: https://github.com/mi3nts/biometricSpeechAnalysis
# ==============================================================================

# import libaries
import spacy
import dash
from dash import html
import dash_table
from dash import dcc
import pandas as pd
import string

text = 'Nevertheless, it remains unclear how peripheral sensory neurons detect visceral osmolality changes, and how they modulate thirst. Here we use optical and electrical recording combined with genetic approaches to visualize osmolality responses from sensory ganglion neurons. Gut hypotonic stimuli activate a dedicated vagal population distinct from mechanical-, hypertonic- or nutrient-sensitive neurons. We demonstrate that hypotonic responses are mediated by vagal afferents innervating the hepatic portal area (HPA), through which most water and nutrients are absorbed. Eliminating sensory inputs from this area selectively abolished hypotonic but not mechanical responses in vagal neurons. Recording from forebrain thirst neurons and behavioural analyses show that HPA-derived osmolality signals are required for feed-forward thirst satiation and drinking termination. Notably, HPA-innervating vagal afferents do not sense osmolality itself. Instead, these responses are mediated partly by vasoactive intestinal peptide secreted after water ingestion. Together, our results reveal visceral hypoosmolality as an important vagal sensory modality, and that intestinal osmolality change is translated into hormonal signals to regulate thirst circuit activity through the HPA pathway.'

nlp = spacy.load("en_core_web_sm")
tokens = nlp(text)

# get features from text - word counts, and part of speech tag counts
pos = {}
words = {}
words_pos = {}

for token in tokens:
    if token.pos_ not in pos:
        pos[token.pos_] = 1
    else:
        cnt = pos[token.pos_]
        pos[token.pos_] = cnt + 1
    
    if token.text not in words and not token.text.isspace() and token.text not in string.punctuation:
        words[token.text] = 1
        words_pos[token.text] = token.pos_
    else:
        if not token.text.isspace() and token.text not in string.punctuation:
            cnt = words[token.text]
            words[token.text] = cnt + 1

# create dataframe from above dictionary
temp = {'Words': list(words_pos.keys()), 'Part of Speech': list(words_pos.values())}
df = pd.DataFrame.from_dict(temp)

# create dash app
app = dash.Dash(__name__)

# create layout for dash app
app.layout = html.Div(children=[
    # part of speech tag counts graph
    dcc.Graph(
        id = 'example-pos-graph',
        
        figure = {
            'data': [
                {'x': list(pos.keys()), 'y': list(pos.values()), 'type': 'bar'}
            ],
            
            'layout': {
                'title': 'Part of Speech Tag Counts',
                'xaxis':{
                    'title':'Part of Speech Tag'
                },
                'yaxis':{
                    'title':'Count'
                }
            }
        }
    ),
    
    # word counts graph
    dcc.Graph(
        id = 'example-words-graph',
        
        figure = {
            'data': [
                {'x': list(words.keys()), 'y': list(words.values()), 'type': 'bar'}
            ],
            
            'layout': {
                'title': 'Word Counts',
                'xaxis':{
                    'title':'Word'
                },
                'yaxis':{
                    'title':'Count'
                }
            }
        }
    ),
    
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True) # run app
