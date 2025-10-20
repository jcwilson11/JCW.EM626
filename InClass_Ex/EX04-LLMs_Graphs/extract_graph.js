// Rhino JavaScript Macro for MagicDraw
// Final version with full semantic edge support and filtering

importPackage(java.io);
importPackage(java.lang);
importPackage(com.nomagic.magicdraw.core);
importPackage(com.nomagic.magicdraw.openapi.uml);
importPackage(com.nomagic.uml2.ext.magicdraw.classes.mdkernel);
importPackage(com.nomagic.uml2.ext.magicdraw.compositestructures.mdinternalstructures);
importPackage(com.nomagic.uml2.ext.magicdraw.activities.mdactivities);
importPackage(com.nomagic.uml2.ext.magicdraw.statemachines.mdbehaviorstatemachines);
importPackage(com.nomagic.uml2.ext.usecases.usecases);
importPackage(com.nomagic.uml2.ext.magicdraw.interactions.mdbasicinteractions);
importPackage(com.nomagic.uml2.ext.sysml.requirements);

// === Setup ===
var fileName = "sysml_graph.xml";
var project = Application.getInstance().getProject();
var root = project.getPrimaryModel();
var outFile = new BufferedWriter(new FileWriter(fileName));
outFile.write("<graph>\n");

var visited = new java.util.HashSet();
var edgeCount = 0;
var edgeTypeCounts = {};

var typeBlacklist = [
  "ProfileApplication", "Slot", "TaggedValue", "ConstraintInstance", "Comment",
  "ElementImport", "PackageMerge", "PackageImport", "EnumerationLiteral",
  "InstanceSpecification", "TemplateBinding", "ValueSpecification"
];

// === Utility Functions ===
function sanitize(value) {
  if (value == null) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function safeDocumentation(element) {
  return element.getDocumentation ? element.getDocumentation() : "";
}

function isNamedSemanticElement(el) {
  var name = el.getName ? el.getName() : "";
  var doc = safeDocumentation(el);
  return name && name.trim() !== "" || (doc && doc.trim() !== "");
}

// === Node and Edge Export ===
function exportNode(element) {
  if (visited.contains(element)) return;
  visited.add(element);

  var type = element.getHumanType ? element.getHumanType() : element.getClass().getSimpleName();
  if (typeBlacklist.indexOf(type) !== -1) return;

  var id = element.getID ? element.getID() : null;
  if (!id) return;

  var name = element.getName ? element.getName() : "Unnamed";
  var doc = safeDocumentation(element);

  if (!isNamedSemanticElement(element) && type === "Element") return;

  outFile.write(
    '  <node id="' + id + '" name="' + sanitize(name) +
    '" type="' + sanitize(type) + '" documentation="' + sanitize(doc) + '" />\n'
  );
}

function exportEdge(source, target, label) {
  if (!source || !target || !label) return;
  if (!isNamedSemanticElement(source) || !isNamedSemanticElement(target)) return;

  try {
    var sid = source.getID();
    var tid = target.getID();
    if (!sid || !tid) return;
    outFile.write('  <edge source="' + sid + '" target="' + tid + '" label="' + sanitize(label) + '" />\n');
    edgeCount++;
    if (!edgeTypeCounts[label]) edgeTypeCounts[label] = 0;
    edgeTypeCounts[label]++;
  } catch (e) {}
}

// === Relationships ===
function handleRelationships(el) {
  var cname = el.getClass().getName();

  if (cname.indexOf("ConnectorImpl") !== -1 && el.getEnd().size() === 2)
    exportEdge(el.getEnd().get(0).getRole(), el.getEnd().get(1).getRole(), "Connector");

  if (cname.indexOf("AssociationImpl") !== -1 && el.getMemberEnd().size() === 2)
    exportEdge(el.getMemberEnd().get(0).getType(), el.getMemberEnd().get(1).getType(), "Association");

  if (cname.indexOf("DependencyImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Dependency");

  if (cname.indexOf("UsageImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Usage");

  if (cname.indexOf("RealizationImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Realization");

  if (cname.indexOf("AbstractionImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Abstraction");

  if (cname.indexOf("SatisfyImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Satisfy");

  if (cname.indexOf("VerifyImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Verify");

  if (cname.indexOf("RefineImpl") !== -1)
    exportEdge(el.getClient().get(0), el.getSupplier().get(0), "Refine");

  if (cname.indexOf("ControlFlowImpl") !== -1 || cname.indexOf("ObjectFlowImpl") !== -1)
    exportEdge(el.getSource(), el.getTarget(), el.getClass().getSimpleName());

  if (cname.indexOf("TransitionImpl") !== -1)
    exportEdge(el.getSource(), el.getTarget(), "Transition");

  if (cname.indexOf("IncludeImpl") !== -1)
    exportEdge(el.getIncludingCase(), el.getAddition(), "Include");

  if (cname.indexOf("ExtendImpl") !== -1)
    exportEdge(el.getExtendedCase(), el.getExtension(), "Extend");

  if (cname.indexOf("InterfaceRealizationImpl") !== -1)
    exportEdge(el.getImplementingClassifier(), el.getContract(), "InterfaceRealization");

  if (cname.indexOf("TestCaseImpl") !== -1 && el.getVerifies) {
    var it = el.getVerifies().iterator();
    while (it.hasNext()) exportEdge(el, it.next(), "TestCaseVerifies");
  }

  if (cname.indexOf("GeneralizationImpl") !== -1)
    exportEdge(el.getSpecific(), el.getGeneral(), "Generalization");

  if (cname.indexOf("InformationFlowImpl") !== -1) {
    var sources = el.getInformationSource(), targets = el.getInformationTarget();
    if (sources && targets) {
      var i, j;
      for (i = 0; i < sources.size(); i++)
        for (j = 0; j < targets.size(); j++)
          exportEdge(sources.get(i), targets.get(j), "InformationFlow");
    }
  }

  if (cname.indexOf("MessageImpl") !== -1)
    exportEdge(el.getSendEvent(), el.getReceiveEvent(), "Message");
}

// === Recursive Visit ===
function visitRecursively(el) {
  var type = el.getHumanType ? el.getHumanType() : el.getClass().getSimpleName();
  if (typeBlacklist.indexOf(type) !== -1) return;

  if (el instanceof NamedElement) exportNode(el);
  handleRelationships(el);

  var owned = el.getOwnedElement();
  if (owned && owned.iterator) {
    var it = owned.iterator();
    while (it.hasNext()) visitRecursively(it.next());
  }
}

// === Run ===
visitRecursively(root);
outFile.write("</graph>\n");
outFile.close();

// === Log Summary ===
Application.getInstance().getGUILog().log("✅ sysml_graph.xml exported successfully.");
Application.getInstance().getGUILog().log("📦 Nodes: " + visited.size());
Application.getInstance().getGUILog().log("🔗 Edges: " + edgeCount);
for (var label in edgeTypeCounts) {
  Application.getInstance().getGUILog().log("🔸 " + label + ": " + edgeTypeCounts[label]);
}
