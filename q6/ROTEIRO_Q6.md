# Roteiro curto - Q6

## 1. Hiperparâmetros

| Parâmetro | Valor |
|---|---:|
| Episódios de treinamento | `1000` |
| Episódios de teste | `100` |
| `learning_rate` | `0.1` |
| `gamma` | `0.99` |
| `epsilon_decay_rate` | `0.001` |
| Janela de convergência | `100` episódios |
| Limiar de convergência | recompensa média `>= -20` |

## 2. Comando

```powershell
python -m q6.experiments --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

O comando executa automaticamente:

1. Q-learning sem shaping;
2. Q-learning com shaping potencial.

## 3. Explicação

A função usada é:

```text
Phi(s) = -distância Manhattan até o objetivo
r' = r + gamma * Phi(s') - Phi(s)
```

Exemplo: se `Phi(s) = -5`, `Phi(s') = -4` e `gamma = 0.99`:

```text
r' = r + 0.99 * (-4) - (-5)
r' = r + 1.04
```

O avanço em direção ao objetivo recebe uma contribuição positiva. O potencial
é aplicado apenas à recompensa de treinamento; o teste deve ser interpretado
com a recompensa original do ambiente.

## 4. Resultados

Registrar:

| Configuração | average_reward | success_rate | Episódios até convergência |
|---|---:|---:|---:|
| Sem shaping | ______ | ______ | ______ |
| Com shaping | ______ | ______ | ______ |

## 5. Análise

Comparar se o shaping aumentou a recompensa média e reduziu o número de
episódios até a convergência. Explicar que o potencial fornece feedback de
progresso mesmo quando a recompensa original é pouco informativa.

Discutir limitações:

- um potencial mal definido pode incentivar atalhos ruins;
- uma escala muito grande pode dominar a recompensa original;
- o resultado depende de `gamma` e dos hiperparâmetros;
- shaping aplicado também no teste pode dificultar a comparação com a métrica
  original.

## 6. Teste rápido

```powershell
python -m unittest q6.test_q6 -v
```
