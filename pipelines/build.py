from pipelines.arcathon_pipeline import ArcathonPipeline, PhindArcathonPipeline, Mixtral87B, LLama2Code


def build_pipeline(name,**kwargs):
    if name =='ArcathonPipeline':
        return ArcathonPipeline(**kwargs)
    elif name =='PhindArcathonPipeline':
        return PhindArcathonPipeline(**kwargs)
    elif name =='Mistral87B':
        return Mixtral87B(**kwargs)
    elif name =='LLama2Code':
        return LLama2Code(**kwargs)