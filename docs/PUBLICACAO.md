# Publicação pendente para revisão

Esta preparação modificou somente a cópia local. Nenhuma ação abaixo foi executada no GitHub; este documento é um roteiro para a etapa que depende de autorização.

## Conferir localmente

```bash
git status --short
git log --graph --oneline --decorate -12
git tag -n3
make cpu test
```

A branch final é `main`. As branches CPU e CUDA preservam blocos de trabalho e as tags anotadas registram seus estados. O antigo `master` permanece como referência do fork. Os documentos locais de experimentos não substituem Issues publicadas.

## Etapas no GitHub após autorização

1. Revisar o histórico e os arquivos incluídos, especialmente resultados, tags e limitações no README.
2. Publicar commits e tags escolhidos, respeitando a branch remota existente e a organização exigida para `main`. Não usar force push para substituir trabalho remoto sem revisão.
3. Publicar os registros de experimento como Issues com as evidências reais; usar os rascunhos locais como ponto de partida. Informar a cronologia real, sem apresentá-los como Issues abertas antes da implementação.
4. Inserir no README os links reais das Issues e tags publicadas. Se houver revisão via PR, registrar o PR real; os merges locais não equivalem a PRs no GitHub.
5. Executar e anexar os experimentos de GPU que ainda faltam, se possível, antes de afirmar ganho entre backends ou conclusão da otimização avançada.
6. Conferir a `main` publicada e o link do repositório a ser entregue no Blackboard. Não submeter uma pasta local ou ZIP como se substituíssem o formato de entrega pedido.

Não foram preparados comandos de push automáticos. As publicações e a submissão final não fazem parte do trabalho local já autorizado.
