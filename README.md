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

O KCRG não tem arestas diretas entre conceitos. Dois conceitos só se encontram por meio de uma categoria ou tag em comum. Para saber se uma recomendação acertou, usamos uma relação que fica fora do grafo, a dos synsets do WordNet anotados na ARASAAC. Dois conceitos formam um par positivo quando compartilham pelo menos um synset, como “tossir” e “tossir no lenço” ou “planta” e “seres vivos”. Os quatro synsets com mais de 50 conceitos ficam de fora, porque juntam todos os números e todas as lojas. Restam 15.719 pares envolvendo 6.295 conceitos. Em 9.518 desses pares os rótulos não têm nenhuma palavra em comum, e eles formam o subconjunto difícil da avaliação.

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
| GraphSAGE | ✓ | ✓ | ✓ |
| GraphSAGE residual | ✓ | ✓ | ✓ |

O MLP e a GraphSAGE usam as mesmas features, o mesmo tamanho de embedding (256) e a mesma perda contrastiva InfoNCE. A perda usa os outros pares do lote e mais 1.024 conceitos sorteados como negativos, com temperatura de 0,05. Por isso a diferença entre os dois mede o efeito do grafo. A GraphSAGE tem duas camadas com agregação por média. A variante residual soma à saída uma projeção da representação do próprio nó antes da passagem de mensagens, para que a identidade textual não se dilua na média dos vizinhos. A variante principal é escolhida pelo MRR no desenvolvimento. O treino usa AdamW com lr de 1e-3 e para cedo pelo MRR no desenvolvimento, com paciência de 20 épocas. Cada modelo é treinado com três seeds.

## Métricas

- Recall@5, Recall@10, NDCG@10 e MRR, calculados por consulta.
- Hits@10 e MRR por par no modo filtrado, em que os outros positivos da consulta saem da frente. Essas duas métricas também são calculadas no subconjunto de pares sem palavra em comum.
- Intervalos de confiança de 95% por bootstrap pareado sobre as consultas de teste, com 10.000 iterações.
- Empates de score são resolvidos por uma permutação fixa.

## Como rodar

```bash
uv sync
uv run python -m gnn.data   # monta data/graph.json e as features (CPU)
uv run python -m gnn.run    # baselines, MLP e GraphSAGE; grava outputs/
```

Tudo roda em CPU. O caminho do PictoInstruct pode ser trocado pela variável `PICTOINSTRUCT_ROOT`.

## Resultados

Teste com 1.327 consultas e 10.772 candidatos. Nos modelos treinados, média e desvio padrão de três seeds. A coluna “Hits@10 difícil” é o Hits@10 por par, filtrado, nos 5.365 pares de teste sem palavra em comum.

| Método | Recall@10 | NDCG@10 | MRR | Hits@10 difícil |
| --- | ---: | ---: | ---: | ---: |
| Aleatório | 0,000 | 0,000 | 0,002 | 0,000 |
| Popularidade | 0,002 | 0,001 | 0,004 | 0,002 |
| Adamic-Adar | 0,251 | 0,213 | 0,290 | 0,209 |
| Cosseno LaBSE | 0,430 | 0,422 | 0,547 | 0,199 |
| Cosseno Serafim | 0,482 | 0,451 | 0,568 | 0,165 |
| MLP | 0,464 ± 0,013 | 0,451 ± 0,006 | 0,568 ± 0,002 | 0,226 ± 0,015 |
| GraphSAGE | 0,452 ± 0,003 | 0,453 ± 0,005 | 0,576 ± 0,008 | 0,312 ± 0,003 |
| GraphSAGE residual | 0,491 ± 0,008 | 0,487 ± 0,006 | 0,607 ± 0,006 | 0,307 ± 0,007 |

A GraphSAGE residual foi escolhida pelo MRR no desenvolvimento (0,614 contra 0,574 da GraphSAGE simples). No bootstrap pareado da seed 3407, ela supera o MLP no MRR por 0,049 (IC 95% de 0,034 a 0,063) e o cosseno Serafim por 0,047 (de 0,029 a 0,064). No Recall@10, a diferença para o Serafim é de 0,015, com intervalo de −0,001 a 0,030, que cruza o zero. O efeito do grafo aparece com mais força nos pares sem palavra em comum, em que as duas GraphSAGE ficam perto de 0,31 de Hits@10, contra 0,23 do MLP e 0,17 do Serafim. Os números completos estão em `outputs/results.json` e os exemplos de recomendação em `outputs/examples.json`.

## Limitações

Compartilhar um synset é um sinal silver de proximidade, e não um julgamento profissional de CAA sobre quais pictogramas devem aparecer juntos numa prancha. Parte dos pares é composicional, como “escovar” e “escovar os dentes”, o que favorece os modelos de texto. O grafo é usado de forma transdutiva, já que os conceitos de teste participam da passagem de mensagens com suas categorias e tags. Isso equivale a supor que um pictograma novo já chega com os metadados da ARASAAC, mas sem anotação de synset.
