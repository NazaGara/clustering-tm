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
  <ul>
    <li><a href="#about-the-project">Proyecto 1: Clustering</a></li>
    <li><a href="#instalacion">Instalacion</a></li>
    <li><a href="#requisitos">Requisitos</a></li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#metodologia">Metodologia</a></li>
    <li><a href="#resultados">Resultados</a></li>
    <li><a href="#licencia">Licencia</a></li>
    <li><a href="#contacto">Contacto</a></li>
  </ul>
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
* Como minima cantidad de veces que los lemas deban aparecer tome el valor 35 (variable MIN_FREQ), ya que me daba alrededor de 200 lemas a clusterizar. 

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
#ejemplo para el part of speech, analogo con tag.
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

Finalmente, agrupo todas las caracteristicas de cada palabra en un unico diccionario:
```python
for token in doc:
  if word_filter(token): continue
  word = token.lemma_
  feats[word] = {**tag[word], **pos[word], **triplas[word], **large_context[word], **close_context[word], **countable[word]}
```

<!--Aca hay que decir lo del countable!-->

Ahora si, con estas 5 caracteristicas ya tenia informacion sobre cada palabra y pase a la siguiente etapa de vectorización.


### Vectorizacion

La parte de vectorización consiste en poder traducir estas caracteristicas que recolectamos en el preproceso en vectores y matrices para poder calcular distancias entre ellas.

Use [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) de sklearn, que mediante un arreglo de diccionarios (tenemos un diccionario por cada palabra en feats[palabra]) nos devuelve una matriz con todas estas caracteristicas ya traducidas. Ademas, voy guardando un diccionario con las palabras que estudiamos y su columna en la matriz.
```python
from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse=False)
features, key_words, wid = [], {}, 0
for word in feats:
  key_words[word] = wid
  wid += 1
  features.append(feats[word])
matrix = vectorizer.fit_transform(X=features)
```

Pero esta matriz obtenida tiene dimensiones realmente grandes, por lo cual queremos normalizar estos vectores y reducir aquellas columnas (es decir, caracteristicas) que no contengan informacion muy especial, por ejemplo, columnas que sean todas nulas o que todas tengan valores similares.

Tambien uso la implementacion de sklearn para poder normalizar y fijarle un limite a la varianza de cada valor:

```python
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
VARIANCE_THRESHOLD = 1e-7
selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
normed_matrix = normalize(matrix, axis=1, norm='l1')
reduced_matrix = selector.fit_transform(normed_matrix)
```

Al final de la etapa de vectorizacion, tenemos una matriz con valores numericos sobre las caracteristicas, y con un tamaño reducido. Esta matriz reducida, sera la que usamos para poder hacer agrupar las palabras y hacer el clustering.

### Clustering

Para hacer el clustering use dos implementaciones del algoritmo:
1. [KMeansClusterer]((https://tedboy.github.io/nlps/generated/generated/nltk.cluster.KMeansClusterer.html)) de ntlk, con distancia coseno
2. [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) de sklearn, con distancia euclidea

Notar que si bien las distancia utilizadas para ambos algoritmos son diferentes, los resultados no varian mucho, ya que como los vectores estan normalizados, ambas distancias son similares.

Implemente dos funciones, tienen los mismos parametros:
* k : la cantidad de clusters a realizar
* matrix: la matriz con los datos reducidos

Ambas retornan en una lista los labels o numero de clusters de cada palabra.

```python
from nltk.cluster import kmeans, cosine_distance
from sklearn.cluster import KMeans

def ntlk_clustering(k, matrix):
  clusterer = kmeans.KMeansClusterer(num_means=k, distance=cosine_distance, avoid_empty_clusters=True)
  clusters = clusterer.cluster(matrix, assign_clusters=True)
  return clusters

def sklearn_clustering(k, matrix):
  clusterer = KMeans(n_clusters=k)
  clusterer.fit(X=matrix)
  return clusterer.labels_
```

Llamo ambas funciones, con numero fijo de clusters. Podemos ver los resultados usando la funcion `show_clusters()`.
```python
NUM_CLUSTERS = 30
ntlk_cluster =  ntlk_clustering(NUM_CLUSTERS, reduced_matrix)
sk_cluster = sklearn_clustering(NUM_CLUSTERS, reduced_matrix)
show_clusters(NUM_CLUSTERS)
```

### Embeddings: LSA y t-SNE


Para aplicar embeddings a los datos obtenidos, me base en la implementacion de sklearn del metodo de LSA (Latent semantic analysis) [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) 

```python
from sklearn.decomposition import TruncatedSVD

def lsa_reduction(matrix):
    svd = TruncatedSVD(n_components=100, n_iter=5)
    lsa_data = svd.fit_transform(X=normed_matrix)
    return lsa_data
```

Para poder verlo visualmente a los clusters, aplicamos el metodo de t-SNE: [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)


```python
from sklearn.manifold import TSNE

def tsne_reduction(matrix):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(reduced_matrix)
    return tsne_data
```

Para poder ver visualmente los datos, creamos un dataframe de Pandas, donde en cargamos la coordenada en x, la coordenada en y (obtenida usando `tsne_reduction`), el valor del cluster y la palabra. Junto con MatplotLib hacemos un scatterplot del dataframe creado.

![clusters](https://imgur.com/7qQApCo.png)

## Resultados

Para poder visualizar los resultados obtenidos, prepare algunas funciones a usar, estas son:
* `show_clusters(k)`
* `cluster_of(word)`
* `plot_cluster_of(word)`
* `same_cluster(word1, word2)`


Con estas funciones, y una lista de palabras `test_words` se ve en pantalla a los clusters correspondiente para cada palabra:
```python
#stopwords1
results['el']
#empleos
results["intendente"]
#numeros
results['mil']
#verbos
results['lograr']
```

Luego de probar diferentes configuraciones de tamaños de ventana y tamaño de clusters, en base a la cantidad de palabras, los "mejores" resultados se obtuvieron con las configuraciones siguientes:

1. close_window = 1, large_window = 7,  #clusters = 40 
1. close_window = 2, large_window = 7,  #clusters = 40
1. close_window = 3, large_window = 15, #clusters = 40

Estos resultados fueron evaluados viendo manualmente los clusters obtenidos de la palabras.

Podemos ver que el procedimiento genera agrupaciones de palabras correctamente, pero que aun se puede mejorar mas. Para hacer que el modelo funcione mejor, habria que seguir probando con las caracteristicas a tomar que resulto en lo que mas influye en los clusters obtenidos.

De aquellas palabras que WIP...

### Trabajo Futuro

Entre las cosas que se no llegue a probar estan:

* Jugar con la matriz obtenida luego de aplicar LSA para hacer clustering.
* Intentar identificar caracteristicas como dimensiones de las matrices.

## Licencia

Distributed under the MIT License. See `LICENSE` for more information.


## Contacto

* [Twitter](https://twitter.com/nazagara99)

* [LinkedIn](https://linkedin.com/in/nazareno-garagiola/)
