import Box from '@mui/material/Box';
import ToolRecommendationCard from "./ToolRecommendationCard";
import {ReactNode, useState} from "react";
import Typography from "@mui/material/Typography";
import {BaseUsage, LLMUsage, ToolUsage, ToolRecommendation} from "@/types/chat";
import SnackbarError from "@/components/Chat/ProgressCards/SnackbarError";
import Collapse from "@mui/material/Collapse";
import ImageProgressCard from "./ImageProgressCard";
import CodeCard from "./CodeCard";
import TextCollapseCard  from "./TextCollapseCard";
import LoggingCard from './LoggingCard';
import {InstallationProgressCard} from "./InstallationProgressCard";

const generate_random_id = () => {
  return Math.random().toString(36).substr(2, 9);
}
const processObjs = (progressJson: any): BaseUsage[] => {
  // check if progressJson is a list, and if its empty
  if (!Array.isArray(progressJson) || !progressJson.length) {
    return [];
  }
  var id_order: string[] = [];
  var obj_dict: any = {};
  var tool_count = 0;
  // create a dictionary of objects, with the block_id as the key
  
  progressJson.forEach((progress) => {
    if (!progress) return [];
    if (progress.block_id === "start") {
    id_order.push("start");
       obj_dict["start"] = (
         <Typography variant="h6" key={generate_random_id()}>Starting request chain...</Typography>
       )
     } else if (progress.block_id === "end") {
       id_order.push("end");
       obj_dict["end"] = (
         <Typography
           variant="h6"
           key={generate_random_id()}
           sx={{
             textAlign: "left"
           }}
         >
           {progress.output}
         </Typography>
       )
     }
    var block_id = progress.block_id;
    if (!id_order.includes(block_id)) {
      id_order.push(block_id);
      if (block_id.includes("tool")) {
        tool_count++;
        obj_dict[block_id] = {
          occurence: tool_count,
          block_id: block_id,
          children: [],
          recommendations: [],
          parent: null,
          ongoing: true
        }
      } else if (block_id.includes("recommendation")) {
        obj_dict[block_id] = {
          occurence: 1,
          block_id: block_id,
          children: [],
          recommendations: [],
          ongoing: true
        }
      } 
    }
    if (block_id.includes("tool")) {
      obj_dict[block_id].depth = 0;
      if (progress.recommendations) obj_dict[block_id].recommendations = progress.recommendations;
      switch (progress.method_name) {
        case "on_agent_action": {
          obj_dict[block_id].api_name = progress.api_name;
          obj_dict[block_id].task = JSON.stringify(progress.task); // progress.action
          obj_dict[block_id].Parameters = JSON.stringify(progress.Parameters); // progress.action_input
          obj_dict[block_id].depth = progress.depth;
          obj_dict[block_id].api_calling = progress.api_calling;
          obj_dict[block_id].task_title = progress.task_title;
          obj_dict[block_id].composite_code = progress.composite_code;
          obj_dict[block_id].ongoing=true;
          break
        }
        case "on_tool_start": {
          obj_dict[block_id].api_name = progress.api_name;
          obj_dict[block_id].api_calling = progress.api_calling;
          obj_dict[block_id].tool_input = progress.tool_input;
          obj_dict[block_id].api_description = JSON.stringify(progress.api_description); // progress.api_description
          break
        }
        case "on_tool_end": {
          // check if progress.output is a string
          if (typeof progress.output === "string") {
            obj_dict[block_id].output = progress.output;
          } else {
            obj_dict[block_id].output = JSON.stringify(progress.output);
          }
          obj_dict[block_id].status = progress.status;
          break
        }
        case "on_agent_end": {
          obj_dict[block_id].ongoing = false;
          break
        }
        default: {
          break
        }
      }
    } else if (block_id.includes("recommendation")) {
      obj_dict[block_id].depth = 0;
      if (progress.recommendations) obj_dict[block_id].recommendations = progress.recommendations;
    } else if (block_id.includes("image")) {
      obj_dict[block_id] = {
        block_id: block_id,
        task: progress.task, 
        children: [],
      };
    } else if (block_id.includes("transfer")) {
      obj_dict[block_id] = {
        block_id: block_id,
        task: progress.task, 
        children: [],
      };
    } else if (block_id.includes("code")) {
      obj_dict[block_id] = {
          occurence: 1,  
          block_id: block_id,
          codeString: progress.task,  
          language: "python",   //progress.task_title
          fullContent: progress.composite_code,
          ongoing: true, 
          children: [],
      }
    } else if (block_id.includes("textcollapse")) {
      obj_dict[block_id] = {
          occurence: 1,  
          block_id: block_id,
          fullContent: progress.task,
          ongoing: true, 
          children: [],
      }
    } else if (block_id.includes("log")) {
      obj_dict[block_id] = {
        occurence: 1,
        block_id: block_id,
        task_title:progress.task_title,
        logString: progress.task,
        tableData: progress.tableData,
        imageData: progress.imageData,
        color:progress.color,
        ongoing: true,
        children: [],
      }
    } else if (block_id.includes("installation")) {
      obj_dict[block_id] = {
        occurence: 1,
        block_id: block_id,
        progress: progress.task_title,
        message: progress.task,
        ongoing: true,
        children: [],
      };
    }
  });
  var ret: BaseUsage[] = [];
  // for each object, add it to the return list
  for (var i = 0; i < id_order.length; i++) {
    var block_id = id_order[i];
    ret.push(obj_dict[block_id]);

  }
  ret = ret.filter((x) => x !== undefined);
  return ret;
}

const generateCards = (progressJson: any) => {
  var progressObjs: BaseUsage[] = processObjs(progressJson);
  var lastdepth = 0;
  var reactNOdeBacktrack = [];
  var startingIndex = 1;
  var root: BaseUsage = {
    occurence: -1,
    type: "root",
    block_id: "root",
    ongoing: false,
    depth: -1,
    children: [],
    parent: null
  }
  var toolRecommendations = []
  for (var i = 0; i < progressObjs.length; i++) {
    if (progressObjs[i].block_id.includes("recommendation")) {
      toolRecommendations.push(
        <ToolRecommendationCard
          key={progressObjs[i].block_id}
          data={progressObjs[i] as ToolRecommendation}
        />
      );
    }
  }
  if (startingIndex >= progressObjs.length) {
    return {
      toolRecommendations: toolRecommendations,
      cards: null,
    }
  }
  var root1 = progressObjs[startingIndex];
  root1.parent = root;
  root.children.push(root1);
  // recursively convert to tree based on depth (progressObjs is the preorder traversal)
  function convertPreorderToTree(index: number, progressObjs: BaseUsage[], root: any) {
    if (index >= progressObjs.length) {
      return;
    }
    var curr = progressObjs[index];
    if (curr.depth > lastdepth) {
      // add to children of last node
      root.children.push(curr);
      curr.parent = root;
      lastdepth = curr.depth;
    } else if (curr.depth == lastdepth) {
      // add to children of parent of last node
      root.parent.children.push(curr);
      curr.parent = root.parent;
    } else {
      // backtrack
      var backtrack = lastdepth - curr.depth;
      for (var i = 0; i < backtrack; i++) {
        root = root.parent;
      }
      root.parent.children.push(curr);
      curr.parent = root.parent;
      lastdepth = curr.depth;
    }
    convertPreorderToTree(index + 1, progressObjs, curr);
  }
  convertPreorderToTree(2, progressObjs, root1);
  let collapsibleComponents: ReactNode[] = [];
  let shouldCollapse = false;
  function convertTreeToReactNodes(root: any): ReactNode {
    var components: ReactNode[] = [];
    if (shouldCollapse) {
      components = collapsibleComponents;
    }
    if (root.block_id === "root") {
      var ret = [];
      for (var i = 0; i < root.children.length; i++) {
        ret.push(convertTreeToReactNodes(root.children[i]));
      }
      return ret;
    }
    if (root.block_id.includes("tool")) {
      var temp2: ToolUsage = root as ToolUsage;
      components.push(
        <Typography variant="body1" key={generate_random_id()}>
          {
            temp2.task && temp2.task.startsWith('"') && temp2.task.endsWith('"')
              ? temp2.task.slice(1, -1)
              : temp2.task
          }
        </Typography>
      )
    } else if (root.block_id.includes("recommendation")) {
      var temp4: ToolRecommendation = root as ToolRecommendation;
      components.push(
        <ToolRecommendationCard
          key={root.block_id}
          data={temp4}
        />
      )
    } else if (root.block_id.includes("transfer")) {
      components.push(
        <ImageProgressCard
          key={generate_random_id()}
          imageSrc={"data:image/png;base64,"+root.task}
        />
      );
    } else if (root.block_id.includes("code")) {
      components.push(
        <CodeCard
          codeString={root.codeString}
          language={root.language}
          fullContent={root.fullContent}
        />
    );
    }  else if (root.block_id.includes("textcollapse")) {
      components.push(
        <TextCollapseCard
          fullContent={root.fullContent}
        />
    );
    } else if (root.block_id.includes("log")) {
      components.push(
        <LoggingCard
          key={generate_random_id()}
          title={root.task_title}
          logString={root.logString}
          tableData={root.tableData}
          logColor={root.color}
          imageData={root.imageData}
        />
      );
    } else if (root.block_id.includes("installation")) {
      components.push(
        <InstallationProgressCard
          progress={root.progress}
          message={root.message}
        />
      );
    }
    for (var i = 0; i < root.children.length; i++) {
      components.push(convertTreeToReactNodes(root.children[i]));
    }
    return (
      <Box sx={{
        my: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        width: "100%",
        '&:hover': {
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          '& > *': {
            backgroundColor: 'transparent',
          },
        }
      }}>
        <Collapse in={!shouldCollapse} timeout="auto" unmountOnExit>
          {components}
        </Collapse>
      </Box>
    )
  }
  var result = convertTreeToReactNodes(root);
  return {
    toolRecommendations: toolRecommendations,
    cards: result,
  }
}

interface ProgressCardControllerProps {
  progressJson: LLMUsage[] | ToolUsage[] | undefined;
  className: string;
}
const renderCollapseButton = (collapsed: boolean) => (
  <Box sx={{ display: 'flex', alignItems: 'center' }}>
    <Typography variant="body2">Finished working</Typography>
    <Box sx={{ width: 40 }}></Box>
    <Typography variant="body2" sx={{ color: 'grey.600' }}>
      {collapsed ? "Hide Work" : "Show Work"} {collapsed ? "↑" : "↓"}
    </Typography>
  </Box>
);

const ProgressCardController = (props: ProgressCardControllerProps) => {
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarContent, setSnackbarContent] = useState("");
  const [collapsed, setCollapsed] = useState(true);
  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  }
  const handleSnackbarOpen = () => {
    setSnackbarOpen(true);
  }
  if (!props.progressJson || props.progressJson.length === 0) {
    return null;
  }
  var temp = generateCards(props.progressJson);
  var cards = temp.cards;
  var toolRecommendations = temp.toolRecommendations;
  return (
    <Box sx={{}}>
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        minWidth: "100%",
        width: "100%"
      }}>
        <Box sx={{ height: '6px' }}></Box>
          {cards}
      </Box>
      <SnackbarError open={snackbarOpen} handleClose={handleSnackbarClose} content={snackbarContent} />
    </Box>
  );
}
export default ProgressCardController;