# synth_local — controle sintético das duas referências

Este dataset existe para responder a **uma** pergunta: o baseline de vizinhança
(peer) é *necessário*, ou é apenas uma forma diferente de olhar a mesma coisa?

O `Synth_unfair` não responde isso. Ele planta um bolsão que destoa da taxa global
**e** da vizinhança ao mesmo tempo, então a SUL e a local z-score acendem juntas e
nada é discriminado. Aqui há **dois** bolsões, distantes um do outro, cada um
isolando uma referência.

## Gabarito

| Bolsão | ρ do bolsão | ρ da vizinhança | ρ global | SUL esperada | local z esperado |
|---|---|---|---|---|---|
| **local** (puramente local) | 0,50 | 0,75 | 0,50 | **≈ 0 — cega por construção** | fortemente negativo |
| **global** (puramente global) | 0,30 | 0,30 | 0,50 | **alta** | **≈ 0 — vizinhos acompanham** |

- O **bolsão local** tem taxa *idêntica* à global, então a razão de
  log-verossimilhança da SUL colapsa: o baseline global não pode vê-lo, não por
  falta de sensibilidade, mas **por construção**. A vizinhança, a 0,75, o denuncia.
- O **bolsão global** destoa da taxa global, mas seus vizinhos destoam junto: a SUL
  acende e a local z-score fica quieta. Isso é o que impede a leitura errada de que
  a local z-score seria "uma SUL mais sensível" — ela responde outra pergunta.

Juntos, os dois provam **complementaridade**, não superioridade.

## Estrutura

13.600 pontos em 17 blobs densos de 800 pontos cada:

- **região local** — 1 blob central (o bolsão, ρ 0,50) cercado por 6 blobs em
  hexágono (ρ 0,75). O hexágono garante que a triangulação de Delaunay torne o
  centro adjacente a todos os 6.
- **região global** — 1 blob central (o bolsão, ρ 0,30) cercado por 4 blobs em
  cruz (ρ 0,30).
- **preenchimento** — 5 blobs distantes (ρ 0,40), calibrados para a taxa global do
  mapa ser **exatamente** 0,50 (6.800 positivos em 13.600 pontos).

Os positivos são atribuídos por **contagem exata** dentro de cada blob (não por
sorteio), então o gabarito vale exatamente para qualquer seed; a seed move apenas
as coordenadas dentro dos blobs.

## Como gerar

Os dados **não são commitados** (convenção de `datasets/README.md`). Gere com:

```bash
uv run python src/synth_data.py --out datasets/synth_local/data --seed 42
```

Depois o dataset atravessa o pipeline como qualquer outro:

```bash
uv run python src/main.py explain --dataset synth_local --min-cluster-frac 0.02 --n-alt-worlds 200 --out outputs
```

A validação automatizada do gabarito vive em `tests/test_synth_data.py` e roda sem
precisar do CSV — ela gera os pontos em memória.
