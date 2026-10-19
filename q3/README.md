# Q3 - Q-learning tabular no MountainCar

O `MountainCar-v0` possui ações discretas, mas seus estados são contínuos:
posição e velocidade. A classe `QLearningAgentTabular` é reutilizada depois de
converter cada estado contínuo para um índice discreto.

## Executar

Na raiz do projeto:

```powershell
python -m q3.experiments --episodes 10000 --test-episodes 100
```

Parâmetros ajustáveis:

```powershell
python -m q3.experiments `
    --episodes 10000 `
    --test-episodes 100 `
    --position-bins 18 `
    --velocity-bins 14 `
    --learning-rate 0.1 `
    --gamma 0.99 `
    --epsilon-decay-rate 0.001
```

Com `18` bins de posição e `14` bins de velocidade, a tabela possui:

```text
18 * 14 = 252 estados
```

Como o MountainCar possui três ações, a tabela Q tem formato `(252, 3)`.

## Testar

```powershell
python -m unittest q3.test_q3 -v
```
