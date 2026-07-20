"""
Rate limiter yapılandırması.
slowapi ile IP tabanlı istek sınırlama.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
