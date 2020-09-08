import xml.etree.ElementTree as ET
import re

assignmentTypes = ['AuditProgram']

singleRoleAssignment = '''<?xml version="1.0" encoding="UTF-8"?>
<openpagesConfiguration>
    <actors/>
    <roleAssignments>
        <roleAssignment status="assign" type="AuditProgram">
            <businessUnits>
                <businessUnit name="/_op_sox/Project/Default/ICDocumentation/Audits/IACE/Plans/G3/G3_S02/G3_S02_M01/2018/AUD2018_Compliance_00000002171"/>
            </businessUnits>
            <roleActors>
                <roleActor name="00000002171"/>
            </roleActors>
            <roles>
                <role name="IACE Audit Object Role"/>
            </roles>
        </roleAssignment>
        <roleAssignment status="assign" type="AuditProgram">
            <businessUnits>
                <businessUnit name="/_op_sox/Project/Default/ICDocumentation/Audits/IACE/Plans/G3/G3_S02/G3_S02_M01/2018/AUD2018_IACE_00000002171"/>
            </businessUnits>
            <roleActors>
                <roleActor name="00000002171"/>
            </roleActors>
            <roles>
                <role name="IACE Audit Object Role"/>
            </roles>
        </roleAssignment>
        <roleAssignment status="assign" type="AuditProgram">
            <businessUnits>
                <businessUnit name="/_op_sox/Project/Default/ICDocumentation/Audits/IACE/Plans/G3/G3_S02/G3_S02_M01/2018/AUD2018_Location_00000002171"/>
            </businessUnits>
            <roleActors>
                <roleActor name="00000002171"/>
            </roleActors>
            <roles>
                <role name="IACE Audit Object Role"/>
            </roles>
        </roleAssignment>
        <roleAssignment status="assign" type="AuditProgram">
            <businessUnits>
                <businessUnit name="/_op_sox/Project/Default/ICDocumentation/Audits/IACE/Plans/G3/G3_S05/G3_S05_M02/2018/AUD2018_IACE_4760011"/>
            </businessUnits>
            <roleActors>
                <roleActor name="5164202"/>
            </roleActors>
            <roles>
                <role name="IACE Audit Object Role"/>
            </roles>
        </roleAssignment>        
    </roleAssignments>
    <administrators/>
    <actorObjectProfileAssociationSets/>
</openpagesConfiguration>
'''

#tree = ET.parse('UPS_Users-op-config.xml')
tree = ET.parse('assignments-op-config.xml')
#tree = ET.ElementTree(element=ET.fromstring(singleRoleAssignment))

#print(type(tree))
root = tree.getroot()
#print(root.tag, root.attrib)

userRoleAssignments = dict()

'''
We have a special case when dealing with assigned domains.
We need to handle this specific case.
        <roleAssignment
                type="SOXBusEntity"
                status="assign">
            <businessUnits>
                <businessUnit
                        name="/_op_sox/Project/Default/BusinessEntity"/>
            </businessUnits>
            <roleActors>
                <roleActor
                        name="OpenPagesAdministrator"/>
            </roleActors>
            <roles>
                <role
                        name="OpenPages Modules 7.0 - All Permissions"/>
            </roles>
        </roleAssignment>

'''
def determineAssginmentType(node):

    if extractDomain(node) != None:
        pass



def extractDomain(node):
    domain = None

    # In this method, we will determine the assginment type.
    # 1. First check for entity type.
    assignmentType = node.get('type')
    if assignmentType == 'SOXBusEntity':
        print("Found Entity assignment")
        # Extract the domain for the Business Entity.
        domain = node.find("./businessUnits/businessUnit").get("name")
        found = re.match('/_op_sox/Project/Default/BusinessEntity(.*)', domain)
        domain = found.group(1)
        print(found.group(1))
    elif assignmentType in assignmentTypes:
        # Check for other objectType assignments.
        print("Found %s assignment." % (assignmentType))
        domain = node.find("./businessUnits/businessUnit").get("name")
        found = re.match('.+/ICDocumentation/Audits(.*)', domain)
        print(found.group(1))
        domain = found.group(1)

    return domain

def getUserRecord(actorName):
    global userRoleAssignments

    if actorName in userRoleAssignments.keys():
        userRecord = userRoleAssignments[actorName]
    else:
        userRecord = createUserRecord(actorName)
        userRoleAssignments[actorName] = userRecord
    return userRecord


def recordRoleAssignment(node):
    global userRoleAssignments

    determineAssginmentType(node)

    actorName = node.find("./roleActors/roleActor").get("name")
    if actorName == None:
        print(node.attrib)

    userRecord = getUserRecord(actorName)

    # Get the assignments for this specific user.
    assignments = userRecord.get('assignments')
    # Append the current assignment
    domain = node.find("./businessUnits/businessUnit").get("name")
    if len(domain) == 0:
        domain = '/' # Because the domain is blank, we assume root.
    assignments.append([domain, node.find("./roles/role").get("name")])
    #print(assignments)
    return

def createUserRecord(actorName):
    newRecord = dict()
    newRecord['name'] = actorName
    newRecord['assignments'] = []
    return newRecord

def displayRoleAssignment(node):
    print('Role Assignemnt')
    print(node.find("./businessUnits/businessUnit").attrib)
    print(node.find("./roleActors/roleActor").attrib)
    print(node.find("./roles/role").attrib)


# for child in root:
#     print(child.tag, child.attrib)
#
# print('-----------------------')
# print(root[1][0].tag)
#
# print('-----------------------')
# # for child in root.iter('roleAssignments'):
# #     print(child.tag, child.attrib)
# #     for roleAssignment in child.iter('roleAssignment'):
# #         print(roleAssignment.tag, roleAssignment.attrib)


# Need to parse all users.


# Parse the role assignments.
print('-----------------------')
for child in root.findall("./roleAssignments/roleAssignment"):
    #displayRoleAssignment(child)
    recordRoleAssignment(child)

print("%s,%s,%s,,,,,%s,%s" % ("Userid", "Domain", "Assigned Role","UserId","Domain"))

actors = list(userRoleAssignments.keys())
actors.sort()
for actorName in actors:
    #print()
    #print(actorName)
    #print(",,,,,,,,")
    for assignments in userRoleAssignments.get(actorName)['assignments']:
        # assignments[0] is the domain of role assignment.
        # assignments[1] is the role assignment.
        # Find AuditProgram
        found = re.match('.+/ICDocumentation/Audits(/.+)|.+/Project/Default/BusinessEntity(/.+)', assignments[0])
        if found:
            if found.group(1):
                print(found.group(1))
            else:
                print(found.group(2))

        # Find Business Entity
        # found = re.match('.+/Project/Default/BusinessEntity(/.+)', assignments[0])
        # if found:
        #     print(assignments[0])


        #print(assignments)
        # print("'%s, %s, %s,,,,,%s,%s" %(actorName, assignments[0], assignments[1],  actorName, ','.join(found.group(1).split('/')) ))

        # AuditProgram assignment
        # print("=\"%s\",%s,%s,,,,,=\"%s\",%s" % (actorName, assignments[0], assignments[1], actorName, ','.join( found.group(1).strip('/').split('/') ) ) )

