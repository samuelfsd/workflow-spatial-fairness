# Referência quantitativa de Sacharidis

Este diretório contém somente entradas versionadas do benchmark. O manifesto registra
valores do artigo *Auditing for Spatial Fairness*, outputs históricos do notebook e os
hashes dos cinco CSVs públicos congelados.

Resultados locais não devem ser escritos aqui. Eles são artefatos regeneráveis da
execução do benchmark.

Cada resultado possui uma identidade estável, a unidade observada, a fonte e a posição
textual da evidência. `precision` diferencia valores exatos, aproximados, arredondados e
outputs históricos salvos. `observation` registra ressalvas necessárias para interpretar
o valor sem alterar sua fonte.

A igualdade de um dataset local é determinada pelo SHA-256 do artefato público. Números
resumidos no artigo são uma comparação adicional e não substituem essa identidade. Isso
é especialmente importante para Crime: o CSV público possui 60.849 casos e TPR de
55,96%, embora o artigo relate 61.266 e aproximadamente 58%.
