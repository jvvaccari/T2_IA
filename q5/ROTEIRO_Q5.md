# Roteiro curto - Q5

## 1. Hiperparâmetros

| Parâmetro | Valor |
|---|---:|
| Episódios de treinamento | `1000` |
| Episódios de teste | `100` |
| `learning_rate` | `0.1` |
| `gamma` | `0.99` |
| `epsilon_decay_rate` | `0.001` |
| `min_epsilon` | `0.01` |
| `max_epsilon` | `1.0` |

## 2. Comandos

Blackjack:

```powershell
python -m q5.experiments --environment blackjack --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

Cliff Walking:

```powershell
python -m q5.experiments --environment cliffwalking --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

Frozen Lake:

```powershell
python -m q5.experiments --environment frozenlake --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

## 3. Explicação

Q-learning é off-policy: atualiza usando a melhor ação possível no próximo
estado:

```text
delta = r + gamma * max Q(s', a') - Q(s, a)
```

SARSA é on-policy: atualiza usando a ação que a política realmente escolheu:

```text
delta = r + gamma * Q(s', a') - Q(s, a)
```

## 4. Resultados

Para cada ambiente, registrar:

| Ambiente | Algoritmo | average_reward | success_rate | Última recompensa |
|---|---|---:|---:|---:|
| Blackjack | Q-learning | ______ | ______ | ______ |
| Blackjack | SARSA | ______ | ______ | ______ |
| Cliff Walking | Q-learning | ______ | ______ | ______ |
| Cliff Walking | SARSA | ______ | ______ | ______ |
| Frozen Lake | Q-learning | ______ | ______ | ______ |
| Frozen Lake | SARSA | ______ | ______ | ______ |

No Cliff Walking, mostrar também uma trajetória de teste de cada agente e
comparar passos, quedas e recompensa acumulada.

## 5. Análise

Discutir:

- se SARSA foi mais conservador ao considerar a exploração real;
- se Q-learning encontrou recompensas maiores ao assumir o melhor próximo
  comportamento;
- qual algoritmo convergiu mais rapidamente;
- qual algoritmo produziu trajetórias mais seguras no Cliff Walking;
- quando SARSA é preferível em ambientes de risco, como robótica;
- quando Q-learning é preferível em simulações e ambientes onde a política
  ótima é o objetivo.

## 6. Teste rápido

```powershell
python -m unittest q5.test_q5 -v
```
