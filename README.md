<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** NazaGara, clustering-tm, twitter_handle, clustering, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->





<!-- PROJECT LOGO -->
<br />
  <p align="center">
    <h1 align="center"> Clustering - Text Mining 2021</h1>
    <h3 align="center"> Garagiola, Nazareno </h3>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Tabla de contenidos</h2></summary>
  <ol>
    <li><a href="#about-the-project">Proyecto 1: Clustering</a></li>
    <li><a href="#instalacion">Instalacion</a></li>
    <li><a href="#requisitos">Requisitos</a></li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#metodologia">Metodologia</a></li>
    <li><a href="#licencia">Licencia</a></li>
    <li><a href="#contacto">Contacto</a></li>
  </ol>
</details>



## Proyecto 1: Clustering

Perteneciente a la materia Mineria de texto 2021 de la licenciatura en Cs. de la Computación de FAMAF - UNC


### Instalacion
Clone the repo:
```sh
git clone https://github.com/NazaGara/clustering-tm.git
```
### Requisitos

Para instalar las librerias necesarias, ejecuta el comando:
```sh
pip install jupyter
pip install -r requirements.txt
```


## Uso

Para utilizarlo, recomiendo subirlo a [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) y ejecutarlo, o por consola ejecutar:

```sh
jupyter-notebook clustering.ipynb
```

## Metodologia

Con el objetivo de poder hacer clustering a las palabras del corpus "lavoztextodump.txt" y basandome en repositorios de años anteriores suministrados por la catedra, la idea para poder trabajar fue la de probar con diferentes combinaciones de caracteristicas, tomando diferentes parametros (como los tamaños de ventana) e identificar cual de estas combinaciones me daba mejores resultados.

Es importante aclarar que se tomaron algunas decisiones de diseño con respecto a que tipo de palabras nos interesaban, estas decisiones son:
* Se toman los primeros 0.5 millones - 2 caracteres del corpus, esto porque coincide con un fin de parrafo y tambien porque tomar un numero mayor enlentecia las pruebas.
* Usamos los lemas de las palabras, esto para simplificar el problema y tener mas cantidad de caracteristicas de cada lema.
* Como minima cantidad de veces que los lemas deban aparecer tome el valor 50 (variable MIN_FREQ), ya que me daba alrededor de 200 lemas a clusterizar. 

Use [Spacy](https://spacy.io/) para poder analizar y estudiar el corpus:
```python
f = open("lavoztextodump.txt", 'r')
text = f.read()[:500000-2] 
#...
nlp = spacy.load("es_core_news_md", vectores=False, entity=False)
```

Spacy nos tokeniza cada palabra del corpus y estos nos permite obtener diferentes atributos sobre cada token, atributos que usaremos para poder sacar carateristicas de las palabras en la etapa de preproceso.

A continuación, explico como fue la exploración de las combinaciones y como cada etapa fue desarrollandose.

### Preproceso

El objetivo para el preproceso del texto, es el de poder recolectar caracteristicas sobre cada palabra a estudiar, que sirvan para poder definir esta palabra.
Primero, contamos la cantidad de lemas para poder definir un minimo de cantidad de palabras a usar:
```python
from collections import Counter
lemmas = []
for token in doc:
  if len(token) > 1 and token.is_alpha:
    lemmas.append(token.lemma_.lower())

counter_lemma = Counter(lemmas)
```

Defino una funcion que uso para poder filtrar las palabras que cumplen el requisitos descritos mas arriba:
```python 
def word_filter(token):
  return (not token.is_alpha) or (token.is_digit) or counter_lemma[token.lemma_] < MIN_FREQ 
```

En las primeras iteraciones del metodo tomamos algunos atributos de los token que procesa Spacy y guardandolos en diferentes diccionarios. Al comienzo, los atributos que tome eran: pos_ (part of speech), tag (fine-grained part of speech)

```python
#ejemplo para el part of speech (pos_)
for token in doc:
  if word_filter(token): continue
  pos[word] = {}
#...
for token in doc:
  if word_filter(token): continue
  if not token.pos_ in pos[word].keys():
    pos[word][token.pos_] = 0
  pos[word][token.pos_] += 1
```

A medida que



Incremente la cantidad de caracteristicas con triplas de dependencias.
```python
  tripla = (f"obj: {token.text} - dep : {token.dep_} - root: {doc[i+1].head.lemma_}")
  if not tripla in triplas[word].keys():
    triplas[word][tripla] = 0
  triplas[word][tripla] += 1
```

Luego, me interesaba poder obtener informacion sobre el contexto de cada palabra. Para esto tenia dos formas de hacerlo:
1. Consistia en tomar un contexto inmediato de cada palabra, viendo los lemmas de las palabras anteriores y posteriores.
2. La otra opcion era la de tomar dos contextos diferentes, uno mas cercano a la palabra objetivo donde solo me interesan palabras que no sean stopwords y otro mas amplio, donde solo las palabras que tambien pasaban el filtro.

Probe ambos metodos, y el segundo traia mejores resultados, ya que tenia dos tamaños de ventana para probar y podia hacer asociaciones de las palabras ignorando las stopwords y buscandole mas significado a cada palabra.

```python
def immediate_related_words(span):
    tokens = list(filter(not_a_stopword, span))
    return list(map(lambda token: token.lemma_, tokens))

def keywords_in(span):
    tokens = list(filter(
        lambda token: not word_filter(token) and not_a_stopword(token), span))
    return list(map(lambda token: token.lemma_, tokens))
```

```python

close_lft, close_rgt = i-close_window, i+close_window
if not (close_lft <= 0 and close_rgt >= len(doc)):
imm_related_words = immediate_related_words(doc[close_lft:close_rgt]) if not token.is_sent_start else immediate_related_words(doc[i:close_rgt])
for w in imm_related_words:
  if w == word: continue
  if not w in close_context[word].keys():
    close_context[word][w] = 0
  close_context[word][w] += 1

large_lft, large_rgt = i-large_window, i+large_window
if not (large_lft <= 0 and large_rgt >= len(doc)):
  keywords_in_context = keywords_in(doc[large_lft: large_rgt]) if not token.is_sent_start else keywords_in(doc[i: large_rgt])
  for w in keywords_in_context:
    if w == word: continue
    if not w in large_context[word].keys():
      large_context[word][w] = 0
    large_context[word][w] += 1

```






### Clustering
WIP

### Obtenido vs Esperado
WIP

### Conclusiones
WIP


## Licencia

Distributed under the MIT License. See `LICENSE` for more information.


## Contacto

* [Twitter](https://twitter.com/nazagara99)

* [LinkedIn](https://linkedin.com/in/nazareno-garagiola/)
