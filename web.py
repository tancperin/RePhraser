import bottle
import rephraser
models = ['ramsrigouthamg/t5_paraphraser', 
          'tancperin/t5-paraphraser-quora', 
          'tancperin/t5-paraphraser-bible']

@bottle.route('/')
def home_page():
    return bottle.template('web/input', models=models)

@bottle.route('/output', method='POST')
def rephrase():
    modelselected = bottle.request.forms.get('model')
    sentence = bottle.request.forms.get('input')
    amount = bottle.request.forms.get('amount')
    seed = bottle.request.forms.get('seed')
    max_length = bottle.request.forms.get('max_length')
    top_k = bottle.request.forms.get('top_k')
    top_p = bottle.request.forms.get('top_p')
    rephrased = rephraser.rephrase(modelselected, sentence, int(amount), float(seed), int(max_length), int(top_k), float(top_p))
    return bottle.template('web/output', rephrased=rephrased, sentence=sentence)


bottle.debug(True)
bottle.run(host='localhost', port=8080)


