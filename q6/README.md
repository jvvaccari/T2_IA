# Q6 - Reward shaping por potencial

O experimento usa o Cliff Walking e compara Q-learning com a recompensa
original e com a recompensa modificada:

```text
r' = r + gamma * Phi(s') - Phi(s)
```

A função potencial usada é:

```text
Phi(s) = -distância Manhattan(s, objetivo)
```

## Executar

```powershell
python -m q6.experiments --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

O programa executa os dois casos e imprime recompensa média, taxa de sucesso e
episódios até a convergência. O limiar atual de convergência é média de
recompensas maior ou igual a `-20` em uma janela de 100 episódios.

## Testar

```powershell
python -m unittest q6.test_q6 -v
```
