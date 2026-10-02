"""Per-dataset Drain settings (regex masks, similarity threshold, depth).

Copied from the logpai/logparser benchmark (MIT licence):
https://github.com/logpai/logparser/blob/main/logparser/Drain/benchmark.py
These were tuned on the LogHub-2k sets themselves, so results using them are
"tuned", not held-out. `generic` mode (default masks) is the honest baseline.
"""

PRESETS = {
    'HDFS': {"regex": ['blk_-?\\d+', '(\\d+\\.){3}\\d+(:\\d+)?'], "st": 0.5, "depth": 4},
    'Hadoop': {"regex": ['(\\d+\\.){3}\\d+'], "st": 0.5, "depth": 4},
    'Spark': {"regex": ['(\\d+\\.){3}\\d+', '\\b[KGTM]?B\\b', '([\\w-]+\\.){2,}[\\w-]+'], "st": 0.5, "depth": 4},
    'Zookeeper': {"regex": ['(/|)(\\d+\\.){3}\\d+(:\\d+)?'], "st": 0.5, "depth": 4},
    'BGL': {"regex": ['core\\.\\d+'], "st": 0.5, "depth": 4},
    'HPC': {"regex": ['=\\d+'], "st": 0.5, "depth": 4},
    'Thunderbird': {"regex": ['(\\d+\\.){3}\\d+'], "st": 0.5, "depth": 4},
    'Windows': {"regex": ['0x.*?\\s'], "st": 0.7, "depth": 5},
    'Linux': {"regex": ['(\\d+\\.){3}\\d+', '\\d{2}:\\d{2}:\\d{2}'], "st": 0.39, "depth": 6},
    'Android': {"regex": ['(/[\\w-]+)+', '([\\w-]+\\.){2,}[\\w-]+', '\\b(\\-?\\+?\\d+)\\b|\\b0[Xx][a-fA-F\\d]+\\b|\\b[a-fA-F\\d]{4,}\\b'], "st": 0.2, "depth": 6},
    'HealthApp': {"regex": [], "st": 0.2, "depth": 4},
    'Apache': {"regex": ['(\\d+\\.){3}\\d+'], "st": 0.5, "depth": 4},
    'Proxifier': {"regex": ['<\\d+\\ssec', '([\\w-]+\\.)+[\\w-]+(:\\d+)?', '\\d{2}:\\d{2}(:\\d{2})*', '[KGTM]B'], "st": 0.6, "depth": 3},
    'OpenSSH': {"regex": ['(\\d+\\.){3}\\d+', '([\\w-]+\\.){2,}[\\w-]+'], "st": 0.6, "depth": 5},
    'OpenStack': {"regex": ['((\\d+\\.){3}\\d+,?)+', '/.+?\\s', '\\d+'], "st": 0.5, "depth": 5},
    'Mac': {"regex": ['([\\w-]+\\.){2,}[\\w-]+'], "st": 0.7, "depth": 6},
}
