# Roteiro curto - Q2

## 1. Hiperparâmetros

Os experimentos usam os seguintes valores padrão:

| Parâmetro | Valor |
|---|---:|
| Episódios de treinamento | `1000` |
| Episódios de teste | `100` |
| `learning_rate` | `0.1` |
| `gamma` | `0.99` |
| `epsilon_decay_rate` | `0.001` |
| `min_epsilon` | `0.01` |
| `max_epsilon` | `1.0` |

O treinamento começa com exploração máxima (`epsilon = 1.0`) e reduz esse
valor gradualmente até se aproximar de `0.01`. Durante o teste, a exploração
fica desativada e o agente escolhe a ação de maior valor Q.

## 2. Blackjack

### Comando

```powershell
python -m q2.experiments --environment blackjack --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

### Apresentar no vídeo

- hiperparâmetros usados;
- recompensa média de treinamento;
- evolução de `epsilon`;
- `average_reward` no teste;
- `success_rate` no teste.

### Análise

Preencher após executar:

```text
Treinamento:
- Recompensa observada: ______
- Epsilon final: ______

Teste:
- average_reward: ______
- success_rate: ______
```

Comentário: o Blackjack é estocástico, portanto a recompensa pode oscilar.
Uma recompensa média negativa ou uma taxa de sucesso baixa indica que 1000
episódios podem não ter sido suficientes para aprender uma política forte.

## 3. Cliff Walking

### Comando

```powershell
python -m q2.experiments --environment cliffwalking --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

### Apresentar no vídeo

- hiperparâmetros usados;
- recompensas e penalidades durante o treinamento;
- evolução de `epsilon`;
- `average_reward` no teste;
- `success_rate` no teste.

### Análise

Preencher após executar:

```text
Treinamento:
- Recompensa observada: ______
- Penalidades observadas: ______
- Epsilon final: ______

Teste:
- average_reward: ______
- success_rate: ______
```

Comentário: recompensas muito negativas indicam que o agente ainda cai no
precipício ou demora para chegar ao objetivo. Uma melhora no `average_reward`
indica que a tabela Q está aprendendo um caminho mais eficiente.

## 4. Frozen Lake

### Comando

```powershell
python -m q2.experiments --environment frozenlake --episodes 1000 --test-episodes 100 --learning-rate 0.1 --gamma 0.99 --epsilon-decay-rate 0.001
```

### Apresentar no vídeo

- hiperparâmetros usados;
- recompensas durante o treinamento;
- evolução de `epsilon`;
- `average_reward` no teste;
- `success_rate` no teste.

### Análise

Preencher após executar:

```text
Treinamento:
- Recompensa observada: ______
- Epsilon final: ______

Teste:
- average_reward: ______
- success_rate: ______
```

Comentário: a recompensa do Frozen Lake é esparsa e o ambiente é
escorregadio. Por isso, uma taxa de sucesso baixa pode ocorrer mesmo após o
treinamento. Uma taxa maior indica que o agente aprendeu a chegar ao objetivo
com mais frequência.

## 5. Comparação final

| Ambiente | average_reward | success_rate | Interpretação |
|---|---:|---:|---|
| Blackjack | ______ | ______ | Resultado influenciado pela aleatoriedade |
| Cliff Walking | ______ | ______ | Penalidades indicam quedas ou caminhos ineficientes |
| Frozen Lake | ______ | ______ | Sucesso depende de recompensa esparsa e escorregamento |

Concluir o vídeo informando:

1. qual ambiente apresentou melhor resultado;
2. qual ambiente foi mais difícil;
3. se 1000 episódios foram suficientes;
4. como os resultados poderiam melhorar aumentando os episódios ou ajustando
   `learning_rate`, `gamma` e `epsilon_decay_rate`.

## 6. Teste rápido antes da gravação

```powershell
python -m unittest q2.test_q2 -v
```
