import json

def convert_to_new_format(results, samples, demo_channel=False):
    """Convert from the old format to the new format."""
    output = {
        "samples": samples,
        "results": {}
    }
    
    # Map default channel to chan-x
    for channel_name, channel_data in results.items():
        channel_key = "chan-x" if channel_name == "default" else channel_name
        
        # Handle different data structures
        if isinstance(channel_data, list):
            # Already in the new format (list of dicts)
            output["results"][channel_key] = channel_data
        else:
            # Old format (dict of method -> values list)
            channel_samples = []
            
            # Add data for each sample and method
            for method_name, method_values in channel_data.items():
                for i, value in enumerate(method_values):
                    # Ensure we have enough sample dicts
                    while i >= len(channel_samples):
                        channel_samples.append({})
                    # Add this method's value to the right sample dict
                    channel_samples[i][method_name] = value
                    
            output["results"][channel_key] = channel_samples
    
    # Add demo channel if requested
    if demo_channel and "chan-y" not in output["results"]:
        output["results"]["chan-y"] = []
        for i in range(len(samples)):
            if i == 0:
                output["results"]["chan-y"].append({"linear": 1.0})
            elif i == len(samples) - 1:
                output["results"]["chan-y"].append({"linear": 1.5})
            else:
                sample_dict = {"cubic": 2.5 + i}
                if i == len(samples) // 2:
                    sample_dict["linear"] = 4.5
                output["results"]["chan-y"].append(sample_dict)
    
    return json.dumps(output, indent=2)
