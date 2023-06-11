pqM = 48
pqDs = 8
pqK = 128
eLen = 1000000

class PQSmolDistModel(nn.Module):
    def __init__(self, subspaceCount, subspaceDim, codewordCount):
        super(PQSmolDistModel, self).__init__()
        self.subspaceCount = subspaceCount
        self.subspaceDim = subspaceDim
        self.codewordCount = codewordCount
    
    def forward(self, query, codewords, embeddings):
        starttime = time.time()
        dtable = torch.linalg.norm(
            codewords - torch.reshape(query, (self.subspaceCount, 1, self.subspaceDim)),
            axis=2
        )
        dists = torch.sum(dtable.reshape(-1)[
            embeddings.to(torch.int32) + torch.arange(self.subspaceCount, dtype=torch.int32).unsqueeze(0) * self.codewordCount
        ], dim=1)
        endtime = time.time()
        print(f"timing {endtime - starttime}")
        return dists


torch.onnx.export(
    PQSmolDistModel(pqM, pqDs, pqK),
    (
        torch.randn(pqM * pqDs),
        torch.randn((pqM, pqK, pqDs)),
        torch.clamp(torch.randn((eLen, pqM)).to(torch.uint8), 0, pqK-1),
    ),
    "original-pq-distance.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['query', 'codebook', 'embeddings'],
    output_names=['output'],
    dynamic_axes={
        "embeddings": {0: "embedding_length"}
    },
    verbose=False,
)
