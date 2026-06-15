person_activity_clases = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_clases)}

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}

activities_labels = {"person": person_activity_labels, "group": group_activity_labels}