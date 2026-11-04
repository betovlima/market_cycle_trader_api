# API v10.8.135 — Retorno da posição x retorno acumulado do Live

## Objetivo

Distinguir claramente o retorno **não realizado da posição atual** (Alpaca `unrealized_plpc`, em fração) do **retorno acumulado da conta durante a operação com a Winner**. Eles não representam a mesma coisa. A conta pode ter diversas compras, vendas ou períodos em CASH ao longo do tempo.

## Origem e metodologia

- A conta continua sendo a Alpaca Live vinculada ao usuário autenticado. O MongoDB do MCT **não guarda capital operacional**, não estabelece saldos financeiros nem cria capital inicial.
- A data inicial é obtida dos registros verificáveis de execução/posição da conta principal ou da autorização OAuth da conta Trader, considerando recebimentos de ordens confirmadas por conta. Não lê decisões de outras contas para calcular patrimônio.
- O endpoint somente leitura `TradingClient.get_portfolio_history(GetPortfolioHistoryRequest(..., timeframe="1D", cashflow_types="ALL"))` fornece `profit_loss` e `profit_loss_pct`, com as categorias de fluxo de caixa da própria Alpaca. **Não utilizar** `equity_atual - equity_inicial` como lucro, porque depósitos/retiradas alteram o patrimônio sem gerar lucro.
- `winner_performance` possui `status`, `return_pct` (fração, ex. 0.25 para +25%), `profit_loss`, `start_date`, `winner_enrolled_date`, `as_of`, `coverage` (complete / from_first_daily_close), `source="alpaca_portfolio_history"` e `scope="broker_account_during_winner"`.
- Se faltar histórico íntegro ou data de início verificável, retornar `status=unavailable`, valores `null` e motivo, jamais 0% inventado. A ausência de `base_value_asof` pode impedir comprovação do P/L do primeiro dia; `coverage=from_first_daily_close` deve ficar visível na tela.
- A API da Alpaca reporta resultado **da conta**. Operações manuais durante esse período podem estar incluídas; portanto, não atribuir todo o retorno exclusivamente à qualidade preditiva da Winner. Não confundir com retorno simulado do TCC.
- Falhas na consulta histórica não afetam a apresentação de saldo, caixa, posições e ordens atuais e **nunca interferem na execução da Winner**.
- A conta principal expõe o resultado em `GET /api/live-market/portfolio` para Admin; Trader vê apenas os valores da própria Alpaca via `GET /api/alpaca-oauth/portfolio`. Não existe fallback para as credenciais do Admin.

## Testes

`tests/test_live_winner_performance.py`: escolha de início por conta, retorno oficial mesmo com aporte, histórico parcial, faltas e erros sem inventar rendimento, nenhuma ordem real. Demais testes Live/OAuth permanecem inalterados.

## Implantação

Branch `feature/live-cumulative-winner-return-20261104`, integrar após CI. Versão API v10.8.135, frontend v10.7.52. Sem novas flags, travas de ordens ou alterações na rotina de trading.
