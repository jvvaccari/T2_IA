# Roteiro curto - Q4

## 1. Hiperparâmetros

| Parâmetro | Valor |
|---|---:|
| Episódios de treinamento | `10000` |
| Episódios de teste | `100` |
| Bins de posição | `18` |
| Bins de velocidade | `14` |
| `learning_rate` | `0.01` |
| `gamma` | `0.99` |
| `epsilon_decay_rate` | `0.001` |
| `min_epsilon` | `0.01` |
| `max_epsilon` | `1.0` |

## 2. Comandos

Features simples:

```powershell
python -m q4.experiments --feature-mode simple --episodes 10000 --test-episodes 100 --position-bins 18 --velocity-bins 14 --learning-rate 0.01 --gamma 0.99 --epsilon-decay-rate 0.001
```

Features refinadas:

```powershell
python -m q4.experiments --feature-mode refined --episodes 10000 --test-episodes 100 --position-bins 18 --velocity-bins 14 --learning-rate 0.01 --gamma 0.99 --epsilon-decay-rate 0.001
```

## 3. O que explicar

O agente não guarda um valor para cada estado visitado. Ele calcula:

```text
Q(s, a) = w · f(s, a)
```

As features simples representam bins de posição, bins de velocidade e ação.
As refinadas adicionam posição e velocidade normalizadas, proximidade do
objetivo, risco de velocidade próxima de zero e interações entre estado e ação.

Os pesos são atualizados pelo erro temporal:

```text
delta = r + gamma * max Q(s', a') - Q(s, a)
w = w + learning_rate * delta * features
```

## 4. Resultados para apresentar

Para cada representação, mostrar:

```text
Feature mode: ______
Tamanho do vetor de features: ______
Norma final dos pesos: ______
average_reward: ______
success_rate: ______
```

## 5. Análise

Comparar as features simples e refinadas com o Q-learning tabular da Q3.
A aproximação linear usa menos memória e pode generalizar para estados não
visitados, mas pode perder detalhes se as features forem insuficientes.

Mais features podem representar melhor proximidade e risco, mas aumentam o
custo, podem dificultar o ajuste dos pesos e podem introduzir sobreajuste.
Uma melhora no `average_reward` e no `success_rate` indica que o refinamento
capturou informações úteis do estado.

| Método | average_reward | success_rate | Observação |
|---|---:|---:|---|
| Q-learning tabular | ______ | ______ | ______ |
| Linear simples | ______ | ______ | ______ |
| Linear refinado | ______ | ______ | ______ |

## 6. Teste rápido

```powershell
python -m unittest q4.test_q4 -v
```
