# gnn-work

**Recomendação de Pictogramas utilizando uma rede neural GNN**

Grupo 2: Lenon Anthony de Souza da Silva, Silas Augusto Ramos dos Santos e William Kauã Soares da Silva.

Dado um conceito de pictograma da ARASAAC, o sistema recomenda os conceitos mais próximos a ele. A recomendação usa o grafo KCRG construído no PictoInstruct, em que cada conceito se liga às categorias e tags que a ARASAAC atribui aos seus pictogramas. Uma GraphSAGE aprende embeddings dos nós a partir do texto do rótulo e da vizinhança no grafo, e os conceitos recomendados são os vizinhos mais próximos nesse espaço.

## Dados

O grafo vem do PictoInstruct (`data/processed/offline/graphs/G_kcrg.json`) e é lido sem alterações.

| Medida | Valor |
| --- | ---: |
| nós | 11.865 |
| conceitos (âncoras) | 10.772 |
| categorias | 567 |
| tags | 526 |
| arestas | 53.135 |

O KCRG não tem arestas diretas entre conceitos. Dois conceitos só se encontram por meio de uma categoria ou tag em comum. Para saber se uma recomendação acertou, usamos uma relação que fica fora do grafo, a dos synsets do WordNet anotados na ARASAAC. Dois conceitos formam um par positivo quando compartilham pelo menos um synset, como “tossir” e “tossir no lenço” ou “planta” e “seres vivos”. Os quatro synsets com mais de 50 conceitos ficam de fora, porque juntam todos os números e todas as lojas. Restam 15.719 pares envolvendo 6.295 conceitos. Em 9.518 desses pares os rótulos não têm nenhuma palavra em comum, e eles formam o subconjunto difícil da avaliação. Os synsets grandes pesam nesse subconjunto, porque 2.091 desses pares vêm de synsets com 20 conceitos ou mais, como letras, províncias espanholas e horas do relógio. Em outros 1.206 pares os dois rótulos são idênticos, já que a ARASAAC tem conceitos diferentes com o mesmo nome.

A partição em otimização, desenvolvimento e teste é a mesma do dataset offline silver do PictoInstruct. O treino usa só os pares em que os dois conceitos estão na otimização. Os conceitos de desenvolvimento e de teste entram como consultas, e todos os 10.772 conceitos concorrem como candidatos.

## Modelos

Cada nó recebe como feature o embedding do rótulo no LaBSE e no Serafim, mais o tipo do nó.

| Modelo | Usa texto | Usa grafo | Treinado |
| --- | :-: | :-: | :-: |
| Aleatório | | | |
| Popularidade (grau no KCRG) | | ✓ | |
| Adamic-Adar (dois saltos no KCRG) | | ✓ | |
| Cosseno LaBSE | ✓ | | |
| Cosseno Serafim | ✓ | | |
| MLP | ✓ | | ✓ |
| GraphSAGE residual sem arestas (controle) | ✓ | | ✓ |
| GraphSAGE | ✓ | ✓ | ✓ |
| GraphSAGE residual | ✓ | ✓ | ✓ |

Os modelos treinados usam as mesmas features, o mesmo tamanho de embedding (256) e a mesma perda contrastiva InfoNCE. A perda usa os outros pares do lote e mais 1.024 conceitos sorteados como negativos, com temperatura de 0,05. A GraphSAGE tem duas camadas com agregação por média. A variante residual soma à saída uma projeção da representação do próprio nó antes da passagem de mensagens. Como o MLP e a GraphSAGE diferem também na arquitetura, o controle é a GraphSAGE residual treinada e avaliada sem nenhuma aresta. Com agregação por média, um nó sem vizinhos recebe zero da passagem de mensagens, então o controle tem a mesma arquitetura e o mesmo número de parâmetros da residual, mas não usa o grafo. A comparação entre a residual e o controle é a mais direta para o efeito do grafo.

A variante principal é escolhida pelo MRR no desenvolvimento. O treino usa AdamW com lr de 1e-3 e para cedo pelo MRR no desenvolvimento, com paciência de 20 épocas. Cada modelo é treinado com três seeds. Os hiperparâmetros foram fixados antes do treino e não passaram por busca.

## Métricas

- Recall@5, Recall@10, NDCG@10 e MRR, calculados por consulta.
- Hits@10 e MRR por par no modo filtrado, em que os outros positivos da consulta saem da frente. Os pares são contados por consulta, então um par entre dois conceitos de teste entra duas vezes, uma em cada sentido.
- As duas métricas por par também são calculadas no subconjunto difícil e, para reduzir o peso dos synsets grandes, nos pares difíceis cujo menor synset em comum tem menos de 20 conceitos.
- Intervalos de confiança de 95% por bootstrap pareado sobre as consultas de teste, com 10.000 iterações. No Hits@10 por par, cada consulta reamostrada pesa pelo número de pares que tem.
- Empates de score são resolvidos por uma permutação fixa.

## Como rodar

```bash
uv sync
uv run python -m gnn.data   # monta data/graph.json e as features (CPU)
uv run python -m gnn.run    # baselines, MLP, controle e GraphSAGE; grava outputs/
```

Tudo roda em CPU. O `gnn.data` lê o PictoInstruct, cujo caminho pode ser trocado pela variável `PICTOINSTRUCT_ROOT`, e baixa os encoders nas revisões fixadas em `gnn/data.py`. O grafo e as features não estão no repositório. O `data/manifest.json` guarda os hashes das fontes e dos arquivos gerados, e `outputs/results.json`, `outputs/examples.json` e `outputs/training_log.json` estão versionados.

## Resultados

Teste com 1.327 consultas e 10.772 candidatos. Nos modelos treinados, média e desvio padrão de três seeds. As duas últimas colunas são o Hits@10 por par, filtrado, nos 5.365 pares de teste sem palavra em comum e nos 3.563 desses pares cujo menor synset tem menos de 20 conceitos.

| Método | Recall@10 | NDCG@10 | MRR | Hits@10 difícil | Hits@10 difícil, synset < 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aleatório | 0,000 | 0,000 | 0,002 | 0,000 | 0,000 |
| Popularidade | 0,002 | 0,001 | 0,004 | 0,002 | 0,003 |
| Adamic-Adar | 0,251 | 0,213 | 0,290 | 0,209 | 0,218 |
| Cosseno LaBSE | 0,430 | 0,422 | 0,547 | 0,199 | 0,198 |
| Cosseno Serafim | 0,482 | 0,451 | 0,568 | 0,165 | 0,231 |
| MLP | 0,465 ± 0,012 | 0,451 ± 0,006 | 0,568 ± 0,003 | 0,226 ± 0,015 | 0,258 ± 0,018 |
| Controle sem arestas | 0,461 ± 0,003 | 0,446 ± 0,003 | 0,559 ± 0,005 | 0,232 ± 0,002 | 0,265 ± 0,003 |
| GraphSAGE | 0,452 ± 0,004 | 0,449 ± 0,006 | 0,573 ± 0,005 | 0,298 ± 0,015 | 0,293 ± 0,004 |
| GraphSAGE residual | 0,494 ± 0,004 | 0,490 ± 0,005 | 0,609 ± 0,006 | 0,312 ± 0,012 | 0,313 ± 0,002 |

A GraphSAGE residual foi escolhida pelo MRR no desenvolvimento (0,614 contra 0,576 da GraphSAGE simples). No bootstrap pareado da seed 3407, ela supera no MRR o controle sem arestas por 0,061 (IC 95% de 0,048 a 0,075), o MLP por 0,049 (de 0,035 a 0,064) e o cosseno Serafim por 0,048 (de 0,031 a 0,064). No Recall@10, a diferença para o Serafim é de 0,015, com um intervalo de −0,000 a 0,030 que inclui o zero. O controle fica um pouco abaixo do MLP no MRR (−0,012, de −0,022 a −0,002), então a arquitetura sem as arestas não explica o ganho da residual.

Nos pares sem palavra em comum, a residual acerta no top 10 31,2% dos pares, contra 23,2% do controle, 22,6% do MLP e 16,5% do Serafim. A diferença para o controle é de 0,078 (de 0,057 a 0,101). Quando o subconjunto se limita aos synsets com menos de 20 conceitos, a vantagem diminui e continua acima de zero, com 0,052 sobre o controle (de 0,039 a 0,067) e 0,083 sobre o Serafim (de 0,060 a 0,105). Nesse recorte, a diferença entre a residual e a GraphSAGE simples inclui o zero (0,016, de −0,003 a 0,035). Os números completos estão em `outputs/results.json` e os exemplos de recomendação em `outputs/examples.json`.

## Limitações

Compartilhar um synset é um sinal silver de proximidade, e não um julgamento profissional de CAA sobre quais pictogramas devem aparecer juntos numa prancha. Parte dos pares é composicional, como “escovar” e “escovar os dentes”, e 1.206 pares têm rótulos idênticos, o que favorece os modelos de texto e também faz rótulos repetidos aparecerem juntos nas listas de recomendação. O grafo é usado de forma transdutiva, já que os conceitos de teste participam da passagem de mensagens com suas categorias e tags. Isso equivale a supor que um pictograma novo já chega com os metadados da ARASAAC, mas sem anotação de synset. Os negativos sorteados no treino vêm de todos os conceitos, inclusive os de desenvolvimento e de teste, e podem incluir parceiros verdadeiros de synset, que a InfoNCE trata como negativos. Nenhum modelo usa os pesos das arestas do KCRG, inclusive o Adamic-Adar.

Os intervalos de confiança vêm de uma única seed e de uma única partição. Duas execuções com as mesmas seeds em CPU não deram números idênticos. A maior diferença foi de 0,027 no Hits@10 difícil de uma seed da GraphSAGE simples, o que é do tamanho da diferença entre as duas GraphSAGE nesse subconjunto. As diferenças para o controle, o MLP e os baselines de texto ficam bem acima dessa variação.
