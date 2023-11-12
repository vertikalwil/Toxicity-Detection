# NLP: Multilanguage Toxicity Detection

Build a deep learning model to detect toxicity in sentences in any language and deploy it on ShinyApps.

Deployment link : https://vertikalwil.shinyapps.io/project_nlp/

### Installation
```
git clone https://github.com/vertikalwil/toxicity-detection.git
cd your-repository
create venv 
activate venv
pip install -r requirements.txt
pip install shiny
shiny run app  
```
Warning: if you want to train the model, make sure you run it on unix environment with CUDA and CUDNN installed to be able to use your GPU.

## Files

For python requirements -- requirements.txt <br>
model -- toxicity_v3.h5 <br>
shiny app -- app.py <br>
exploration file -- Exploration.ipynb

















