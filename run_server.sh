until $(th computePatchProposalServer.lua); do
    echo "Sharpmask server crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
