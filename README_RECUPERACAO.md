# Recuperação da administração MongoDB — Market Cycle Trader

Origem confirmada: `market_cycle_trader_api_v1_12_15_strategy_configuration_api`.

## Endpoints recuperados

Todos exigem o header:

`X-Parameter-Bootstrap-Token: <PARAMETER_BOOTSTRAP_API_TOKEN>`

### Bootstrap e status

- `GET /api/admin/parameters/status`
- `POST /api/admin/parameters/bootstrap`

O bootstrap cria documentos ausentes, repara uma estratégia incompatível com o schema atual, arquiva o documento anterior e preserva configurações válidas já administradas pela API.

### Administração completa da estratégia

- `GET /api/admin/strategy-configuration`
- `PATCH /api/admin/strategy-configuration`
- `PUT /api/admin/strategy-configuration`
- `POST /api/admin/strategy-configuration/reset`
- `GET /api/admin/strategy-configuration/history?limit=50`
- `POST /api/admin/strategy-configuration/history/{history_id}/restore`

As alterações:

- validam o documento com `BacktestRequest`;
- são bloqueadas quando existe backtest `queued` ou `running`;
- usam `expected_revision` para evitar atualização concorrente;
- arquivam a versão anterior em `backtest_settings_history`;
- incrementam `revision`;
- recalculam o status da configuração sem reiniciar a API.

### Inicialização administrativa

- `GET /api/admin/setup/status`
- `POST /api/admin/setup/initialize`

A inicialização combina bootstrap de parâmetros, vinculação da conta Alpaca Paper e criação do estado isolado, preservando documentos válidos existentes.

## Scripts MongoDB recuperados

- `scripts/bootstrap_parameters.py`
- `scripts/apply_locked_config.py`
- `scripts/export_locked_config.py`
- `scripts/bootstrap_market_history.py`
- `scripts/migrate_xgboost_only_v1_12_0.py`
- `scripts/apply_paper_trading_config.py`

## Comandos principais

Consultar o estado:

```powershell
python scripts/bootstrap_parameters.py --status
```

Executar bootstrap/migração:

```powershell
python scripts/bootstrap_parameters.py
```

Validar uma configuração sem gravar:

```powershell
python scripts/apply_locked_config.py C:\CAMINHO\config.json --dry-run
```

Aplicar uma configuração completa:

```powershell
python scripts/apply_locked_config.py C:\CAMINHO\config.json `
  --name "xgboost-high-performance-seed-3042" `
  --note "restore validated optimized configuration"
```

Exportar o documento ativo:

```powershell
python scripts/export_locked_config.py C:\CAMINHO\active_config.json
```

## Integração

No `main.py`, os routers eram registrados com:

```python
application.include_router(parameter_bootstrap.router)
application.include_router(strategy_configuration.router)
application.include_router(admin_setup.router)
```

A configuração canônica privada não foi incluída neste ZIP. Ela foi separada em outro artefato para não publicar os parâmetros da estratégia junto ao código.

## Observação financeira

Os arquivos recuperados identificam a configuração como `xgboost_high_performance_seed_3042`. O histórico pesquisável confirma o campeão documentado de US$ 32.589,77 e identifica o seed 3042 como a variante de maior desempenho escolhida para operação. Não foi encontrado, nos relatórios textuais preservados, um comprovante exato do valor final próximo de US$ 40 mil; esse valor pode ter vindo de uma execução posterior não incluída nos relatórios indexados.
