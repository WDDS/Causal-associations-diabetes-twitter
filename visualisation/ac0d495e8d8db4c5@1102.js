// https://observablehq.com/@adahne/cause-and-effect-associations-in-diabetes-related-tweets@1102
export default function define(runtime, observer) {
  const main = runtime.module();
  const fileAttachments = new Map([["cause_effect_nodes_links_reduced_wo_diabetes_10_minNedges_10@4.json",new URL("./files/f0a50029581bc528b2e605233bae6b93f5d65c7005e9c1c124c322655b268f82f9965c752a4be05aa80726f8328ddc5c1d47380cb49dfd2077f1fba2e6680e05",import.meta.url)],["cause_effect_nodes_links_reduced_diabetes_10_minNedges_10@4.json",new URL("./files/2582b2b9969d7c31d91dbb9a1374ea046e06fe127cf09ada95cbbd0ceb416345834a116d126e37b5dacdb3aaa1f099dcd52d97ca7697c2dd42e1e182a8b41f06",import.meta.url)]]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], function(md){return(
md`# Cause and effect associations in diabetes related Tweets



**Adrian Ahne**, adrian.ahne@protonmail.com`
)});
  main.variable(observer()).define(["md"], function(md){return(
md`## Context

This visualisation is part of a study aiming to identify cause-effect associations in diabetes related tweets. Following steps have been conducted to determine causes and effects:
- *Personal* and *non-jokes* tweets: We were only interested in studying tweets containing personal information such as emotions, opinion and *personal* content and in consequence *institutional* tweets (advertising, health studies, etc.) were removed. Similarly jokes related to diabetes is a frequent phenomenon and were removed as well. Both personal and non-jokes tweets were obtained by fine-tuning transformer based language models
- A special focus of our study lied on diabetes distress which regroups all psychological factors related to the day-to-day disease management such as anxiety, fear, concerns, emotions. Only tweets containing an emotional element, either emoji/emoticon or emotional word were kept.
- Fine-tuning a BERTweet based language model allowed us to focus only on sentences containing causal information
- A conditional random field algorithm in combination with BERTweet features enabled us to detect causes and effects
- The identified causes and effects in diabetes-related tweets are visualised in the following

`
)});
  main.variable(observer()).define(["md"], function(md){return(
md `## Visualization: `
)});
  main.variable(observer()).define(["md"], function(md){return(
md` 
The causes (source) and effects (target) have been extracted from diabetes-related Twitter data.
`
)});
  main.variable(observer()).define(["html"], function(html)
{
  const table = html`
  <table style="width:100%">
    <tr>
      <th>Attribute</th>
      <th>Type of Attribute</th> 
      <th>Ordering direction</th>
    </tr>
    <tr>
      <td>Links.source</td>
      <td>Categorical</td> 
      <td>-</td>
    </tr>
    <tr>
      <td>Links.target</td>
      <td>Categorical</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Links.value</td>
      <td>Ordered Quantitative</td>
      <td>Sequential</td>
    </tr>
    <tr>
      <td>Node.name</td>
      <td>Categorical</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Node.group</td>
      <td>Categorical</td>
      <td>-</td>
    </tr>
  </table>
  <br>`;
  return table;
}
);
  main.variable(observer()).define(["md"], function(md){return(
md `## Code`
)});
  main.variable(observer("viewof minNnode")).define("viewof minNnode", ["Inputs"], function(Inputs){return(
Inputs.range([10, 1000], {label: "Min number of causes/effects per node"})
)});
  main.variable(observer("minNnode")).define("minNnode", ["Generators", "viewof minNnode"], (G, _) => G.input(_));
  main.variable(observer("viewof minNedge")).define("viewof minNedge", ["Inputs"], function(Inputs){return(
Inputs.range([10, 1000], {label: "Min number of cause-effect associations per edge"})
)});
  main.variable(observer("minNedge")).define("minNedge", ["Generators", "viewof minNedge"], (G, _) => G.input(_));
  main.variable(observer("chart")).define("chart", ["drawChart","data_wo_diab"], function(drawChart,data_wo_diab){return(
drawChart(data_wo_diab)
)});
  main.variable(observer()).define(["data_wo_diab"], function(data_wo_diab){return(
data_wo_diab.nodes.filter(d => d.parentName == "Insulin")[0].parentClass
)});
  main.variable(observer()).define(["md"], function(md){return(
md `#### Network Graph`
)});
  main.variable(observer("drawChart")).define("drawChart", ["minNnode","minNedge","d3","width","height","color","drag","invalidation"], function(minNnode,minNedge,d3,width,height,color,drag,invalidation){return(
function drawChart(data) {
  const nodesFiltered = data.nodes.map(d => Object.create(d))
      .filter(d => d.nodesize >= minNnode)  // filter nodes with minimal number of causes/effects 
  const links = data.links.map(d => Object.create(d))
      .filter(d => nodesFiltered.map(node => node.name).includes(d.source) && nodesFiltered.map(node => node.name).includes(d.target))
      .filter(d => d.value >= minNedge); // filter edges with min number of associations
  // don't show nodes that are not linked to any other node (when filtered)
  const nodes = nodesFiltered.filter(node => links.map(link => link.source).includes(node.name) || links.map(link => link.target).includes(node.name));

  const maxNlinks = Math.max.apply(null, data.links.map(d => Object.create(d)).map(d => d.value))
  const radius = 7;
  
  var tooltip = d3.select("body")
	.append("div")
	.attr("class", "tooltip")
	.style("opacity", 0);
  
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.name))
    .force("charge", d3.forceManyBody().strength(-800)) // adjust to separate nodes more
    .force("center", d3.forceCenter(width/2, height/2))
    .force('collision', d3.forceCollide().radius(d => d.radius));

  const svg = d3.create("svg")
    .attr("viewBox", [0, 0, width, height])

  const link = svg.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(links)
    .join("line")
      .attr("class", "link")
      .attr("stroke-width", d => Math.max.apply(null, [1, 2*d.value / maxNlinks]) ) // Math.sqrt(d.value) / 2) // play on arrow size
      .attr("marker-end", "url(#end)"); // marker-end: defines arrowhead that will be drawn at the final vertex of the given shape: "url(#arrowhead)"

link.append("title").text(d => d.value);

  // build the arrow
  const arrows = svg.append("svg:defs").selectAll("marker")  
      // let's add two markers: one for unhovered links and one for hovered links.
      .data(["end", "end-active"]) // Different link/path types can be defined here  
      .enter().append("svg:marker")
      .attr("id", String) // String; "arrowhead"
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)  // sets offset of the arrow from the centre of the circle; If circle is bigger, the value needs to be bigger
      //.attr("refY", -1.5)
      .attr("markerWidth", 6) // set the bounding box for the marker
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .attr("xoverflow", "visible")
      .append("svg:path")
         .attr("d", "M0, -5L10,0L0,5")
         .attr("class", d => d=== "end-active" ? "active arrowPath" : "arrowPath");

  //define the classes for each of the markers.
  svg.append("svg:defs").select("#end").attr("class", "arrow");
  svg.append("svg:defs").select("#end-active").attr("class", "arrow-active");

  const node = svg.append("g")
    .attr("class", "nodes")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
      .attr("r", radius)
      .attr("fill", d => color(d))
      .call(drag(simulation))
  	.on('mouseover.fade', fade(0.2))
  	.on('mouseout.fade', fade(1));

  // text for the nodes
  const textNode = svg.append('g')
    .selectAll('.node-label')
    .data(nodes)
    .join('text')
      .text(d => d.name)
      .attr("class", "node-label")
      .attr('font-size',10)
      .call(drag(simulation))
    .on('mouseover.fade', fade(0.2))
  	.on('mouseout.fade', fade(1));

  // text for edges 
  const textLine = svg.append("g")
    .selectAll(".edge-label")
    .data(links)
    .join("text")
      .text(d => d.value)
      .attr("class", "edge-label")
      .attr("font-size", 10)
      .call(drag(simulation))
      .on('mouseover.fade', fade(0.2))
  	  .on('mouseout.fade', fade(1));

  // runs the animation of the force layout one "step". Those steps give the force layout its fluid movement
  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    
    node
      .attr("cx", function(d) { return d.x = Math.max((radius+1), Math.min(width - (radius+1), d.x)); })
      .attr("cy", function(d) { return d.y = Math.max((radius+1), Math.min(height - (radius+1), d.y)); });
    textNode
      .attr("x", d => d.x + 10)
      .attr("y", d => d.y)
      .attr("visibility", "hidden");
    textLine
      .attr("visibility", "hidden")
    	    .attr("x", function(d) {
	        if (d.target.x > d.source.x) { return (d.source.x + 0.99*(d.target.x - d.source.x)/2); }
	        else { return (d.target.x + 0.99*(d.source.x - d.target.x)/2); }
	    })
	    .attr("y", function(d) {
	        if (d.target.y > d.source.y) { return (d.source.y + (d.target.y - d.source.y)/2); }
	        else { return (d.target.y + (d.source.y - d.target.y)/2); }
      })
  });

  // elements fade on touch
  function fade(opacity) {
    return d => {
      node.style('opacity', function (o) { return isConnected(d, o) ? 1 : opacity });
      node.attr("r", function(o) {return isConnected(d,o) ? radius+1 : radius});
      textNode.style('visibility', function (o) { return isConnected(d, o) ? "visible" : "hidden" });
      textLine.style('visibility', o => (o.source === d || o.target === d ? "visible" : "hidden"));
      link.style('stroke-opacity', o => (o.source === d || o.target === d ? 1 : opacity));
      link.attr("marker-end", o => o.source.index == d.index || o.target.index == d.index ? "url(#end-active)" : "url(#end)");


      // move out of node
      if(opacity === 1){
        node.style('opacity', 1)
        node.attr("r", radius)
        node.select("circle").attr("r", radius);
        textNode.style('visibility', 'hidden');
        textLine.style("visibility", "hidden");
        link.style('stroke-opacity', 0.3)
        link.attr("marker-end", "url(#end)");
        //arrows.style("fill-opacity", 0.3)
      }
    };
  }

var unique_parent_names=nodes.map(d => d.parentName).filter((v, i, a) => a.indexOf(v) === i);
var legend = svg.selectAll(".legend")
    .data(unique_parent_names) 
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

legend.append("rect")
    .attr("x", width - 15)
    .attr("width", 15)
    .attr("height", 15)
    .style("fill", function(d) {return color(nodes.filter(n => n.parentName === d)[0]);});

legend.append("text")
    .attr("x", width - 20)
    .attr("y", 9)
    .attr("dy", ".25em")
    .style("text-anchor", "end")
    .style("font-size", "14px")
    .text(function(d) { return d; });
  
  const linkedByIndex = {};
  links.forEach(d => {
    linkedByIndex[`${d.source.index},${d.target.index}`] = 1;
  });

  function isConnected(a, b) {
    return linkedByIndex[`${a.index},${b.index}`] || linkedByIndex[`${b.index},${a.index}`] || a.index === b.index;
  }

  function isLinkForNode(node, link){
	return link.source.index == node.index || link.target.index == node.index;
}
  
  invalidation.then(() => simulation.stop());
  
  return svg.node()
}
)});
  main.variable(observer()).define(["html"], function(html){return(
html`<style>

  .active.arrowPath {
    fill-opacity:1;
  }

  .arrowPath {
    fill-opacity:0.2;
  }



.node {
  stroke: #fff;
  stroke-width: 1.5px;
  opacity: 0.2;
}

.node-active{
  stroke: #555;
  stroke-width: 1.5px;
  opacity: 1;
}

.link {
  stroke: #555;
  stroke-opacity: .2;
}

.link-active {
  stroke: #555;
  stroke-opacity: 1;
}

.links {
  stroke: #aaa;
  stroke-opacity: 0.2;
}

.nodes {
  stroke: #000;
}
  </style>`
)});
  main.variable(observer("height")).define("height", function()
{return 700; console.log("DD");}
);
  main.variable(observer()).define(["md"], function(md){return(
md `#### Drag Function`
)});
  main.variable(observer("drag")).define("drag", ["d3"], function(d3){return(
simulation => {
  
  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  
  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }
  
  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  
  return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
}
)});
  main.variable(observer()).define(["md"], function(md){return(
md `#### Color Code`
)});
  main.variable(observer("color")).define("color", ["d3"], function(d3)
{
  const scale = d3.scaleOrdinal(d3.schemeSet2);
  return d => scale(d.parentClass)
}
);
  main.variable(observer()).define(["md"], function(md){return(
md `## Imports`
)});
  main.variable(observer()).define(["md"], function(md){return(
md `#### Data Import`
)});
  main.variable(observer("data_diab")).define("data_diab", ["FileAttachment"], function(FileAttachment){return(
FileAttachment("cause_effect_nodes_links_reduced_diabetes_10_minNedges_10@4.json").json()
)});
  main.variable(observer("data_wo_diab")).define("data_wo_diab", ["FileAttachment"], function(FileAttachment){return(
FileAttachment("cause_effect_nodes_links_reduced_wo_diabetes_10_minNedges_10@4.json").json()
)});
  main.variable(observer()).define(["md"], function(md){return(
md `#### D3 Import`
)});
  main.variable(observer("d3")).define("d3", ["require"], function(require){return(
require("d3@5")
)});
  main.variable(observer()).define(["md"], function(md){return(
md`## MIT License

### Copyright (c) 2021 Adrian Ahne.
<br>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`
)});
  return main;
}
