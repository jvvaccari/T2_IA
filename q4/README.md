# Q4 - Q-learning com aproximação linear

## Executar

Representação simples:

```powershell
python -m q4.experiments --feature-mode simple --episodes 10000 --test-episodes 100 --position-bins 18 --velocity-bins 14 --learning-rate 0.01 --gamma 0.99 --epsilon-decay-rate 0.001
```

Representação refinada:

```powershell
python -m q4.experiments --feature-mode refined --episodes 10000 --test-episodes 100 --position-bins 18 --velocity-bins 14 --learning-rate 0.01 --gamma 0.99 --epsilon-decay-rate 0.001
```

As features simples usam posição discretizada, velocidade discretizada e ação.
As refinadas adicionam posição/velocidade contínuas, proximidade do objetivo,
risco de velocidade baixa e interações entre bins e ação.

## Testes

```powershell
python -m unittest q4.test_q4 -v
```
