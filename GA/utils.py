def interpolate(value,start,end,mapped_start,mapped_end):
    ori_span = end - start
    mapped_span = mapped_end - mapped_start

    valued_scaled = float(value-start)/float(ori_span)

    return mapped_start + (mapped_span * valued_scaled)