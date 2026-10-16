
from getpass import getpass
from alpaca.trading.client import TradingClient

# Credenciais LIVE, digitadas localmente
api_key = "AKPZWOVVM3LCH2R4CGKXGKD3QZ"
secret_key = "AZDhtRe6zMwbrzks21VUR214p2DnyoyJPvtDFX2hqNHh"

# Conexão à corretora real
client = TradingClient(
    api_key.strip(),
    secret_key.strip(),
    paper=False
)

# Apenas consultas, sem envio de ordens
account = client.get_account()

print("Status da conta:", account.status)
print("Bloqueio de negociação:", account.trading_blocked)
print("Saldo disponível:", account.cash)

positions = client.get_all_positions()
print("Quantidade de posições:", len(positions))

for position in positions:
    print(position.symbol, position.qty)



account = client.get_account()
print("Account ID:", account.id)
