import json

def convert_to_new_format(results, samples, demo_channel=False):
    """Convert from the old format to the new format.
    
    In this updated version, we don't include the interpolation method in the output
    since there is only one value per sample per channel regardless of the method
    used to generate it.
    """
    output = {
        "samples": samples,
        "results": {}
    }
    
    # Map default channel to chan-x
    for channel_name, channel_data in results.items():
        channel_key = "chan-x" if channel_name == "default" else channel_name
        
        # Initialize values array for this channel
        output["results"][channel_key] = []
        
        # Handle different data structures
        if isinstance(channel_data, list):
            # Already in the new format (list of dicts)
            for sample_dict in channel_data:
                # Just take the first value from each sample's methods dictionary
                if sample_dict:
                    # Get the first method and its value
                    method_name = next(iter(sample_dict))
                    value = sample_dict[method_name]
                    output["results"][channel_key].append(value)
                else:
                    # Empty sample, use None
                    output["results"][channel_key].append(None)
        else:
            # Old format (dict of method -> values list)
            # Check if we have any methods
            if channel_data:
                # Just use the first method's values
                method_name = next(iter(channel_data))
                method_values = channel_data[method_name]
                # Add each value directly
                for value in method_values:
                    output["results"][channel_key].append(value)
    
    # Add demo channel if requested
    if demo_channel and "chan-y" not in output["results"]:
        output["results"]["chan-y"] = []
        for i in range(len(samples)):
            if i == 0:
                output["results"]["chan-y"].append(1.0)
            elif i == len(samples) - 1:
                output["results"]["chan-y"].append(1.5)
            else:
                output["results"]["chan-y"].append(2.5 + i)
    
    return json.dumps(output, indent=2)
