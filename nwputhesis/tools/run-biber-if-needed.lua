local raw_doc = arg[1]
if not raw_doc or raw_doc == "" then
  os.exit(0)
end

local function normalize_doc(value)
  local doc = value:gsub("\\", "/")
  doc = doc:gsub("^%./", "")
  doc = doc:gsub("/+$", "")
  doc = doc:gsub("%.tex$", "")

  local name = doc:match("([^/]+)$") or doc
  return doc, name
end

local doc, jobname = normalize_doc(raw_doc)
local aux = doc .. ".aux"
local bbl = doc .. ".bbl"

local function file_exists(path)
  local f = io.open(path, "r")
  if f then
    f:close()
    return true
  end
  return false
end

local function read_file(path)
  local f = io.open(path, "r")
  if not f then
    return nil
  end
  local content = f:read("*a")
  f:close()
  return content
end

local function remove_file(path)
  if file_exists(path) then
    os.remove(path)
  end
end

local function shell_quote(value)
  return '"' .. value:gsub('"', '\\"') .. '"'
end

if not file_exists(aux) then
  remove_file(bbl)
  print("Skip biber: no aux file.")
  os.exit(0)
end

local aux_content = read_file(aux) or ""
if not aux_content:find("\\abx@aux@cite", 1, true) and not aux_content:find("\\citation{", 1, true) then
  remove_file(bbl)
  print("Skip biber: no citations.")
  os.exit(0)
end

local command = "biber " .. shell_quote(jobname)
local ok, _, code = os.execute(command)
if ok == true then
  os.exit(0)
end

if type(ok) == "number" then
  os.exit(ok)
end

os.exit(code or 1)
