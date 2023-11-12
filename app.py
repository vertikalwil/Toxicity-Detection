from shiny import App, render, ui, reactive
import shinyswatch
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from googletrans import Translator
from iso639 import Lang

app_ui = ui.page_fluid(
    shinyswatch.theme.darkly(),
    ui.panel_title("NLP Project: Toxicity Detection"),
    ui.input_text_area(
        "input_sentence",
        "Input Sentence:",
        placeholder="Input any sentence in any language and it will check for toxicity, example : I will shoot you / eu vou atirar em você",
        height="110px",
        width="380px"
    ),
    ui.input_action_button("check", "Detect!", class_="btn-primary"),
    ui.br(),
    ui.br(),
    ui.output_text("lang_output"),
    ui.output_text("output"),
    align='center',
    style="padding: 100px"
)


def server(input, output, session):

    ui.modal_show(
            ui.modal("Please be patient, as it may take a while for detection to come out when using this web app for the first time.", 
             title="Notes", 
             easy_close=False)
    )

    df = pd.read_csv('train.csv')
    vector = TextVectorization(max_tokens=200000, output_sequence_length=1800, output_mode='int')
    vector.adapt(df['comment_text'].values)
    model = tf.keras.models.load_model('toxicity_v3.h5')
    text = ""
    res = []

    @reactive.Effect
    @reactive.event(input.check)
    def get_sentence():
        nonlocal text
        text = input.input_sentence()
        if text == "":
            text = "pxz"
            ui.modal_show(
                ui.modal("Please input something", 
                    title="Warning", 
                    easy_close=False)
            )
        
    @reactive.Effect
    @reactive.event(input.check)
    def process():
        nonlocal text, vector, model, res

        translator = Translator()
        en = translator.translate(text).text
        input = vector([en])

        results = model.predict(input)

        res = []
        for idx, col in enumerate(df.columns[2:]):
            if results[0][idx] > 0.4:
                res.append(col)

    @render.text
    @reactive.event(input.check)
    def lang_output():
        nonlocal text
        langu = Translator().detect(text).lang
        if text == 'pxz':
            return 'Language detected : -'
        return f'Language detected : {Lang(langu).name}'
    
    @render.text
    @reactive.event(input.check)
    def output():
        nonlocal res
        if len(res) > 0:
            words = ', '.join(map(str, [word.upper() for word in res]))
            return f'Toxicity connotations are detected within the following categories: {words}'
        else:
            return 'No toxicity connotations are detected'


app = App(app_ui, server)
