from lambeq.rewriter import RemoveCupsRewriter

class DiagramRewriter: 
    def __init__(self):
        self.rewrite = RemoveCupsRewriter()

    def rewrite_diagram(self , diagram):
        return self.rewrite(diagram)

    def rewrite_batch(self , diagrams):
        return [self.rewrite_diagram(d) for d in diagrams]



