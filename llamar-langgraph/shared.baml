// BAML uses Jinja2, which has a `join` filter that can replicate
// the 'join_conjunction' logic.
// This file defines common data structures and reusable prompt templates.

// A class to hold all environment constants, making them easy to pass around.
class EnvConstants {
  grid_width int
  grid_height int
  fire_types string[]
  extinguish_types string[]
  amt_fire_types int
  inventory_types string[]
  min_required_agents int
  inventory_capacity int
  all_intensities string[] // e.g. ["None", "Low", "Medium", "High"]
  critical_intensity string // e.g. "High"
  cardinal_directions string[]
  l_to_m int
  m_to_h int
}

// A template for the detailed environment description (replaces ENV_STR).
template_string PrintEnvDetails(env: EnvConstants) #"
The environment consists of fires and lost persons, along with reservoirs, deposits, and robots (you). All in a grid with width {{ env.grid_width }} and height {{ env.grid_height }}.

Initially, the robots can see all the fires, but does not know the location of any of the lost people - robots must explore.
The fires can be of {{ env.amt_fire_types }} different types: {{ env.fire_types | join(' or ') }}, each requiring a different resource to extinguish - {{ env.extinguish_types | join(' and ') }} respectively. Make sure you use the proper resource to do so.
A fire consists of a group of 'flammable' objects with intensities of {{ env.all_intensities | join(' or ') }}. It is divided into different regions geographically, so all regions that aren't extinguished (intensity {{ env.all_intensities[0] }}), must be properly addressed before fire can be extinguished. The first few regions (1,2,etc) are the sources of the fire, and must be addressed first.
At each step, if a flammable object has an intensity of {{ env.all_intensities[1] }} or higher, it'll increase in intensity if not extinguished. They spread quickly, so it's important for almost all robots to work collectively to stop the fire.
In {{ env.l_to_m }} steps, the flammable object will go from {{ env.all_intensities[1] }} to {{ env.all_intensities[2] }}. In {{ env.m_to_h }} steps, the flammable object will go from {{ env.all_intensities[2] }} to {{ env.all_intensities[3] }}.
Once a flammable object reaches an intensity of {{ env.critical_intensity }} and not before, it spreads to its immediate neighboors (neighboors with intensity {{ env.all_intensities[0] }} start with intensity of {{ env.all_intensities[1] }}).  In order to extinguish a fire, a robot can use the appropriate extinguish resource at that location.
Then, all the flammable objects in or immediately around this location will lower in intensity by one notch (e.g. from {{ env.all_intensities[2] }} to {{ env.all_intensities[1] }}, or {{ env.all_intensities[3] }} to {{ env.all_intensities[2] }}).

The reservoirs can be of type {{ env.extinguish_types | join(' or ') }}, resources can only be collected at a rate of 1-unit per step.
Thus, to get more resources, you have to collect the resources multiple times.

The deposits can hold any amount and type of resources: {{ env.inventory_types | join(' and ') }}.
The robots can store their entire inventory into the deposit in order to save it for other robots to use.
When a robot gets a certain resource type (if any is available) from a deposit, the space left in their inventory is filled with that resource type.
Deposits create an unnecessary middle-step when using them to store resources, so do not waste time in this way.

If any robot enters the area of visibility of a lost person, that person is found and all robots can now see it.
Once a person is found, at least {{ env.min_required_agents }} robots are required to carry it (could be more depending on person). Otherwise, the person cannot be moved.
A carried person should be dropped into a deposit (any suffices).
To drop a carried person, all agents should have navigated to deposit, and they must ALL perform the DropOff action.

The robots have an inventory capacity of {{ env.inventory_capacity }} with slots for {{ env.inventory_types | join(' and ') }}.
"#

// A template for the observation description (replaces OBS_STR).
template_string PrintObserverInfo(agent_names: string[]) #"
You will get a description of the task robots are supposed to do. You will get an textual description of the environment from the perspective of {{ agent_names | join(' and ') }} as the observation input. You will also get a list of objects each robot is able to see in the environment. Here the objects will have a distinct name which will also include which type of object it is.
So, along with the observation inputs you will get the following information:
"#

// A template for the important details in the Actor prompt (replaces DETAILS_STR).
template_string PrintActorDetails(env: EnvConstants) #"
Important details described below:
    * Even if the robot can see an object, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigates to the objects they want to interact with.
    * When navigating to fire, please specify which specific region of the fire you wish to target.
    * Additionally, when you use the supply, it will be dropped wherever you are, NOT in the region you said. So, make sure to navigate to wherever you wish to use a supply.
    * When a fire has an average intensity of {{ env.all_intensities[0] }}, it means all the flammable objects have been extinguished completely.
    * When the person is not initially visible, you must Explore.
    * When a person is carried, no other action other than navigation and "DropOff" can be made by any of the robots carrying it.
    * When a robot carries a person, all their other resources are dropped and the person takes the entire inventory space.
    * When a group robot wants to drop a person, they must all navigate to the deposit strictly before dropping them. Then, they should ALL perform the DropOff action.
    * When a robot is successful in carrying a person, that just means that specific robot is carrying it, but the person might still not be moveable if an insufficient amount of robots is carrying it.
    * A fire is divided into different regions, and the fire itself (which is just the center). However, there might be flammable objects from the fire that aren't immediately neighbooring this location, so the robot might have to move in different directions to reach them.
    * The resources in the inventory can only be used on fires one unit at a time, so do multiple UseSupply until you clear our your inventory.
    * When a robot is doing a "Move(<direction>)" action, if there is an obstacle, they will not be able to move in that direction.
"#