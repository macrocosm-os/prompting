import subprocess

coldkey = 'sn1-test'
netuid = 61
network = 'test'

miners = [
    # Mock (static)
    {'hotkey':'m1', 'port':9001, 'file':'neurons/miners/test/mock.py', 'type':'mock'},
    {'hotkey':'m2', 'port':9002, 'file':'neurons/miners/test/mock.py', 'type':'mock'},
    # Echo
    {'hotkey':'m3', 'port':9003, 'file':'neurons/miners/test/echo.py', 'type':'echo'},
    {'hotkey':'m4', 'port':9004, 'file':'neurons/miners/test/echo.py', 'type':'echo'},
    {'hotkey':'m5', 'port':9005, 'file':'neurons/miners/test/echo.py', 'type':'echo'},
    # Phrase
    {'hotkey':'m6', 'port':9006, 'file':'neurons/miners/test/phrase.py', 'type':'phrase',   'config': '--neuron.phrase "That is an excellent question"'},
    {'hotkey':'m7', 'port':9007, 'file':'neurons/miners/test/phrase.py', 'type':'phrase',   'config': '--neuron.phrase "Could you repeat that?"'},
    {'hotkey':'m8', 'port':9008, 'file':'neurons/miners/test/phrase.py', 'type':'phrase',   'config': '--neuron.phrase "And so it goes..."'},
    {'hotkey':'m9', 'port':9009,
     'file':'neurons/miners/test/phrase.py', 'type':'phrase',   'config': '--neuron.phrase "You and me baby ain\'t nothing but mammals"'},
    {'hotkey':'m10', 'port':9010 , 'file':'neurons/miners/test/phrase.py', 'type':'phrase', 'config': '--neuron.phrase "I\'m sorry Dave, I\'m afraid I can\'t do that"'},
]

validators = [
    {'hotkey':'v1', 'port':9000 , 'file':'neurons/validator.py', 'type':'real', 'config': '--neuron.sample_size 5 --neuron.device cuda'},        
    {'hotkey':'v2', 'port':9011 , 'file':'neurons/validator.py', 'type':'mock', 'config': '--neuron.sample_size 5 --neuron.model_id mock'},
]

neurons = miners + validators

for neuron in neurons:

    # Construct the PM2 start command
    command = f"pm2 start {neuron['file']} --interpreter python3 --name {neuron['hotkey']}:{neuron['type']} --"\
            +f" --wallet.name {coldkey} --wallet.hotkey {neuron['hotkey']} --subtensor.network {network} --netuid {netuid}"\
            +f" --axon.port {neuron['port']} --axon.external_port {neuron['port']} --logging.debug {neuron.get('config')}"
    print(command)
    subprocess.run(command, shell=True)
