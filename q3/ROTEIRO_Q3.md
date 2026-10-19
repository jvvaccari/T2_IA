# Roteiro curto - Q3

## 1. Hiperparâmetros

Usar inicialmente:

| Parâmetro | Valor |
|---|---:|
| Episódios de treinamento | `10000` |
| Episódios de teste | `100` |
| Bins de posição | `18` |
| Bins de velocidade | `14` |
| `learning_rate` | `0.1` |
| `gamma` | `0.99` |
| `epsilon_decay_rate` | `0.001` |
| `min_epsilon` | `0.01` |
| `max_epsilon` | `1.0` |

## 2. Comando

Na raiz do projeto:

```powershell
python -m q3.experiments --episodes 10000 --test-episodes 100 --position-bins 18 --velocity-bins 14 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

## 3. O que explicar no vídeo

- O estado original possui duas variáveis contínuas: posição e velocidade.
- Cada intervalo contínuo é dividido em bins.
- A posição vira um índice entre `0` e `17`.
- A velocidade vira um índice entre `0` e `13`.
- O índice final é calculado por:

```text
state_id = position_id * 14 + velocity_id
```

- A tabela Q resultante possui `18 * 14 = 252` estados e `3` ações.
- O agente tabular da Q2 é reutilizado sem modificar a regra de atualização.

## 4. Resultados

Mostrar durante a execução:

- formato da tabela Q;
- recompensa durante o treinamento;
- valor final de `epsilon`;
- `average_reward` no teste;
- `success_rate` no teste.

Preencher:

```text
Treinamento:
- Recompensa observada: ______
- Epsilon final: ______

Teste:
- average_reward: ______
- success_rate: ______
```

## 5. Análise

Uma recompensa próxima de `-200` indica que o episódio terminou pelo limite de
passos sem alcançar o objetivo. Uma recompensa menos negativa e uma taxa de
sucesso maior indicam que a discretização e o treinamento aprenderam uma
política melhor.

Se a taxa de sucesso permanecer baixa, testar mais episódios, mais bins ou
outros valores de `learning_rate`, `gamma` e `epsilon_decay_rate`. Mais bins
representam melhor o estado contínuo, mas aumentam o tamanho da tabela e a
quantidade de experiências necessárias.

## 6. Teste rápido

```powershell
python -m unittest q3.test_q3 -v
```
