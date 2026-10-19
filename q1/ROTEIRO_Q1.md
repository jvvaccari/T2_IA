# Roteiro curto - Q1

## 1. Agentes e parâmetros

O jogo Quarto possui dois agentes:

| Agente | Parâmetro utilizado |
|---|---|
| Minimax | Profundidade `2` |
| MCTS | `500` iterações |

O Minimax utiliza poda alfa-beta e uma heurística que:

- retorna `1000` para vitória do jogador avaliado;
- retorna `-1000` para vitória do adversário;
- adiciona `10` pontos para cada linha com três peças que compartilham um
  atributo.

O MCTS utiliza:

1. seleção pelo UCB1;
2. expansão de ações não testadas;
3. simulação aleatória até o fim da partida;
4. retropropagação do resultado.

## 2. Executar contra o Minimax

Na raiz do projeto, executar:

```powershell
python q1\main.py
```

Quando aparecer:

```text
1. Minimax (Profundidade 2)
2. MCTS (500 iterações)
Opção:
```

Digite:

```text
1
```

Durante o vídeo, mostrar:

- o tabuleiro inicial;
- a peça recebida pelo jogador;
- a escolha da posição;
- a peça entregue ao agente;
- a resposta do Minimax;
- o resultado final.

### Análise do Minimax

Registrar:

```text
Resultado: ______
Tempo aproximado das jogadas: ______
Comportamento observado: ______
```

Comentário sugerido: o Minimax é mais previsível porque avalia
deterministicamente as jogadas até profundidade 2. Sua principal limitação é
enxergar poucas jogadas à frente devido à grande quantidade de ações possíveis
no Quarto.

## 3. Executar contra o MCTS

Executar novamente:

```powershell
python q1\main.py
```

Na tela de escolha, digitar:

```text
2
```

Durante o vídeo, mostrar:

- o tabuleiro inicial;
- algumas decisões do agente;
- a peça posicionada;
- a peça entregue ao jogador;
- o resultado final.

### Análise do MCTS

Registrar:

```text
Resultado: ______
Tempo aproximado das jogadas: ______
Comportamento observado: ______
```

Comentário sugerido: o MCTS explora possibilidades por meio de 500 simulações
e escolhe a ação mais visitada. Como as simulações são aleatórias, suas
decisões podem variar entre partidas.

## 4. Comparação final

| Critério | Minimax | MCTS |
|---|---|---|
| Estratégia | Busca com profundidade limitada | Simulações estatísticas |
| Avaliação | Heurística de linhas promissoras | Resultado das simulações |
| Parâmetro | Profundidade `2` | `500` iterações |
| Comportamento | Mais previsível | Mais variável |
| Limitação | Busca curta | Simulação aleatória |

Preencher após as partidas:

| Agente | Resultado | Tempo observado | Observação |
|---|---|---|---|
| Minimax | ______ | ______ | ______ |
| MCTS | ______ | ______ | ______ |

Concluir o vídeo informando:

1. qual agente apresentou melhor resultado nas partidas observadas;
2. qual agente tomou decisões mais rapidamente;
3. qual agente foi mais consistente;
4. como aumentar a profundidade do Minimax ou as iterações do MCTS poderia
   melhorar as jogadas;
5. que o MCTS poderia usar simulações guiadas, em vez de apenas aleatórias.

## 5. Comandos de validação

Para verificar a sintaxe dos módulos:

```powershell
python -m compileall -q q1
```

Para iniciar a partida:

```powershell
python q1\main.py
```

Também é possível executar como pacote:

```powershell
python -m q1.main
```
