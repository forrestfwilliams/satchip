def test_hyp3_satchip(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_satchip', '-h'])
    assert ret.success
