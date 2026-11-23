# Migração CUDA e hipótese de redução de lançamentos

O treino e a inferência foram portados integralmente para CUDA, preservando a rede, a sequência de SGD, o estado inicial e as contas do legado. A atualização entre amostras não foi paralelizada: cada amostra usa os pesos produzidos pela anterior.

## Hipóteses implementadas

1. `cuda-naive`: um lançamento por amostra, parâmetros globais, inferência com uma thread por imagem.
2. `cuda-global`: um lançamento por época e parâmetros globais; isola parte do custo de lançamentos em relação à versão inicial.
3. `cuda-shared`: um lançamento para todas as épocas e parâmetros compartilhados no bloco. Muda simultaneamente a quantidade de lançamentos e a região de memória; não isola perfeitamente os dois efeitos.

O treino usa um bloco cooperativo. Na inferência global/compartilhada, cada bloco reutiliza os pesos entre duas, quatro ou oito imagens, conforme o tamanho total de 64, 128 ou 256 threads. Threads sem imagem válida continuam participando das barreiras do bloco.

## Evidência disponível

- Compilação com nvcc 12.8.93 para `sm_75` e PTX.
- Mesma lógica dos kernels exercitada em CPU com threads e barreiras, incluindo blocos 64/128/256 e último grupo de 17 imagens. Não simula races, bancos ou escalonamento real da GPU.
- Em 23/11, uma execução real de `cuda-shared`, bloco 128, completou 51 épocas e uma inferência de 1797 imagens, com 1717 acertos.
- Perdas por época, 17970 escores, previsões, matriz de confusão e 2778 parâmetros coincidiram exatamente com as referências locais, sem executar novamente o legado no cluster.

Tempos da execução única: treino de 0.768943237 s; inferência de 0.000064663 s; preparação de 1.711880743 s, incluindo H2D. D2H de 0.034912 ms. Veja [registro e limites](../../results/cluster/minimo/RELATO.md) e [JSON](../../results/cluster/minimo/cuda-shared.json).

## Interpretação e pendências

A observação comprova funcionamento da configuração executada; não comprova speedup entre backends ou frente à CPU. Não há baseline e repetições equivalentes na mesma GPU. Contexto e transferências têm custo relevante em um treino pequeno, e um único bloco limita o paralelismo do treino.

Ainda faltam Compute Sanitizer real, Nsight Systems/Compute, medições dos outros backends e escolha experimental do bloco. A bateria completa foi preparada, mas não executada após o pedido de coleta mínima. Perfis ou resultados ausentes não foram preenchidos com estimativas.

As tolerâncias do comparador permanecem `atol=1e-6`, `rtol=1e-5`, com previsões e matriz de confusão idênticas. Não foram ampliadas para produzir aprovação. A [Issue #3](https://github.com/gretafg/awful-barenn/issues/3) registra a migração funcional e foi publicada retrospectivamente. Sua [cópia](../issues/02-cuda.md) acompanha este repositório; os experimentos GPU pendentes permanecem identificados.
