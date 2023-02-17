from flask_caching import Cache
"""
Cache um Laufzeit zu reduzieren
"""
cache = Cache(config={'CACHE_TYPE': 'SimpleCache',
                      'CACHE_DEFAULT_TIMEOUT': 300,
                      'CACHE_THRESHOLD': 500})
