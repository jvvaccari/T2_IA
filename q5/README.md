# Q5 - Q-learning versus SARSA

## Executar as comparações

```powershell
python -m q5.experiments --environment blackjack --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

```powershell
python -m q5.experiments --environment cliffwalking --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

```powershell
python -m q5.experiments --environment frozenlake --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

O comando treina os dois agentes com os mesmos hiperparâmetros e imprime
`average_reward`, `success_rate` e a recompensa do último episódio de treino.

## Diferença dos algoritmos

Q-learning é off-policy e usa:

```text
target = r + gamma * max_a' Q(s', a')
```

SARSA é on-policy e usa a ação realmente escolhida pela política:

```text
target = r + gamma * Q(s', a')
```

No Cliff Walking, as trajetórias de teste também são registradas para
comparar caminhos e penalidades.

## Testes

```powershell
python -m unittest q5.test_q5 -v
```
