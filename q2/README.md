# Q2 - Q-Learning tabular

## Ambientes

Os experimentos usam:

- `Blackjack-v1`, com `natural=False` e `sab=False`;
- `CliffWalking-v1`;
- `FrozenLake-v1`, mapa `4x4` e `is_slippery=True`.

## Executar um experimento

Na raiz do projeto:

```powershell
python -m q2.experiments --environment blackjack --episodes 1000 --test-episodes 100
python -m q2.experiments --environment cliffwalking --episodes 1000 --test-episodes 100
python -m q2.experiments --environment frozenlake --episodes 1000 --test-episodes 100
```

Os parâmetros podem ser alterados com `--learning-rate`, `--gamma` e
`--epsilon-decay-rate`.

## Testes rápidos

```powershell
python -m unittest q2.test_q2
```
