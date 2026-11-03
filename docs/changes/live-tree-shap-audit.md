# Auditoria TreeSHAP das decisões LIVE

Esta mudança é **exclusivamente diagnóstica** e não altera os 52 indicadores
do Control, os modelos treinados, a utilidade prevista, a margem de troca ou
qualquer chamada de negociação da Alpaca.

## Coleta da decisão

Depois que o runtime U67/U77 calcula as utilidades com os modelos LightGBM
finais, chama `model.predict(feature_row, pred_contrib=True)` usando a
**mesma linha de indicadores da decisão**. A saída nativa do LightGBM contém
52 contribuições TreeSHAP mais o termo constante (baseline).

A auditoria aceita uma explicação somente quando:

- os 52 valores das entradas e das contribuições são finitos;
- a dimensão é exatamente 53, incluindo o baseline;
- a soma `baseline + sum(contribuições)` reproduz a utilidade **já calculada**
  pelo runtime, dentro da tolerância numérica de `1e-6 * max(1, |utilidade|)`.

Uma falha de explicação não modifica a decisão, e nenhum modelo é retreinado
para gerar dados da auditoria. O MongoDB recebe o resultado junto do registro
`live_winner_session_runs`, separado por conta Live e sessão.

## Exportação

O botão **Exportar auditoria** continua usando `start_date` e `end_date`.
O ZIP mantém `live_portfolio.json`, `live_decisions_period.csv`,
`live_utility_scores_period.csv`, `audit_period.json` e `README.txt`.

Em consultas por período, acrescenta-se
`live_feature_contributions_period.csv`, em formato de uma linha por
`decision_date` + `execution_session` + `symbol`, contendo:

- `method`, `base_value`, `prediction`, `contribution_sum`,
  `reconstructed_prediction` e `reconstruction_error`;
- 52 colunas `value_<feature>`, correspondendo às entradas reais;
- 52 colunas `shap_<feature>`, correspondendo à influência de cada entrada.

Exemplo de interpretação: para entender CAPR em 02/11, filtre
`symbol=CAPR` e `decision_date=2026-11-02`, ordene as colunas `shap_*`
por valor absoluto e represente valores positivos e negativos num gráfico.
A soma de todos os `shap_*`, adicionada ao baseline, deve reproduzir
`prediction`.

## Limitações e segurança

- **Somente decisões futuras**, a partir da implantação, terão SHAP gravado
  automaticamente. A exportação não cria explicações retroativas nem
  transforma decisões não salvas em dados fictícios.
- Decisões anteriores mantêm os dados de ranking, e o CSV de SHAP pode ter
  apenas cabeçalho. `audit_period.json` registra a cobertura.
- Uma contribuição SHAP mostra a explicação local do **modelo**, não um
  efeito causal ou garantia de retorno.
- A API só consulta a conta Alpaca Live autenticada para filtrar o histórico;
  dados de conta, credenciais e ordens não são gravados no novo CSV.
- O serviço de exportação permanece somente leitura; não executa ordens.
