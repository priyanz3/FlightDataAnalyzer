#-------------------------------------------------------------------------------
# Dependency Tree
# ===============
def ordered_set(alist):
    """
    Creates an ordered set from a list of tuples or other hashable items
    TODO: Move to library
    """
    mmap = {} # implements hashed lookup
    oset = [] # storage for set
    for item in alist:
        #Save unique items in input order
        if item not in mmap:
            mmap[item] = 1
            oset.append(item)
    return oset

def dependency_tree(nodes, app):
    """ Returns a Graph Breadth First Search across a tree of dependencies
    ref: http://en.wikipedia.org/wiki/Breadth-first_search
    
    @param nodes: [obj, obj]
    @param app: []
    """
    node_list = []
    def traverse_tree(app):
        if isinstance(app, str):
            # recorded raw parameter
            return True #end of this branch
        elif not app.dependencies:
            # derived param without children. end of this branch
            if app.recorded():
                return True
            else:
                return False
        #print [node['name'] for node in app['parents']]
        for node_name in app.dependencies:
            node_list.append(node_name)
        dependencies_available = []
        for node_name in app.dependencies:
            try:
                node = nodes[node_name]
            except KeyError:
                # node unavailable
                continue
            active = traverse_tree(node)
            if active:
                dependencies_available.append(node_name)
            else:
                continue
        if app.can_operate(dependencies_available):
            return True
        else:
            return False
    traverse_tree(app)
    return ordered_set(reversed(node_list)) #REVERSE?

#-------------------------------------------------------------------------------



# Instantiate
# ===========    
nodes = {
    SAT : Sat(SAT),
    MACH : Mach(MACH),
    MAX_MACH_CRUISE : MaxMachCruise(MAX_MACH_CRUISE),
    }

# raw parameters recorded on this frame
recorded_params = (TAT, IAS, ALT)
for param_name in recorded_params:
    nodes[param_name] = param_name ##type(param_name, (object,), {})
    
# what we need at the end
app = Derived('TOP_NODE')
app.dependencies = (MAX_MACH_CRUISE, )
# result
process_order = dependency_tree(nodes, app)

assert len(process_order) == 6
# assert dependencies are met
assert process_order.index(TAT) < process_order.index(SAT)
assert process_order.index(ALT) < process_order.index(SAT)
assert process_order == [ALT, TAT, SAT, IAS, MACH, MAX_MACH_CRUISE]

# check we can still process MACH without SAT (uses TAT only)
##del nodes[SAT]
##process_order = dependency_tree(nodes, app)
##assert process_order == [ALT, IAS, TAT, MACH, MAX_MACH_CRUISE]

# without IAS, nothing can work
del nodes[IAS]
process_order = dependency_tree(nodes, app)
assert process_order == []

print "finished tests!"