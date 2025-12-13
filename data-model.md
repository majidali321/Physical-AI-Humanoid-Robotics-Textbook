# Data Model for Physical AI & Humanoid Robotics Textbook

## Content Structure

### Course Structure
- **Course** (Physical AI & Humanoid Robotics)
  - **Modules**: 4 main modules
  - **Weeks**: 13 weeks total
  - **Topics**: Specific topics per week
  - **Pages**: Individual content pages

### Module Entity
```
{
  "id": "string",
  "name": "string",
  "description": "string",
  "weeks": ["Week"],
  "learningObjectives": ["string"],
  "prerequisites": ["string"]
}
```

### Week Entity
```
{
  "id": "string",
  "moduleId": "string",
  "weekNumber": "integer",
  "title": "string",
  "topics": ["Topic"],
  "pages": ["Page"],
  "learningObjectives": ["string"]
}
```

### Topic Entity
```
{
  "id": "string",
  "weekId": "string",
  "title": "string",
  "description": "string",
  "content": "markdown",
  "resources": ["Resource"],
  "exercises": ["Exercise"]
}
```

### Page Entity
```
{
  "id": "string",
  "topicId": "string",
  "title": "string",
  "content": "markdown",
  "author": "string",
  "lastModified": "date",
  "relatedPages": ["string"]
}
```

### Resource Entity
```
{
  "id": "string",
  "type": "enum(video|article|paper|code|diagram)",
  "title": "string",
  "url": "string",
  "description": "string",
  "topicId": "string"
}
```

### Exercise Entity
```
{
  "id": "string",
  "topicId": "string",
  "title": "string",
  "description": "string",
  "difficulty": "enum(easy|medium|hard)",
  "solution": "string",
  "codeTemplate": "string"
}
```

## Content Relationships

### Module-Week Relationship
- One Module contains Many Weeks
- Each Week belongs to exactly one Module

### Week-Topic Relationship
- One Week contains Many Topics
- Each Topic belongs to exactly one Week

### Topic-Page Relationship
- One Topic contains Many Pages
- Each Page belongs to exactly one Topic

### Topic-Resource Relationship
- One Topic contains Many Resources
- Each Resource belongs to exactly one Topic

### Topic-Exercise Relationship
- One Topic contains Many Exercises
- Each Exercise belongs to exactly one Topic

## Validation Rules

### Module Validation
- Name must be unique
- Description must be provided
- Module must have 1-4 weeks
- Learning objectives must be specified

### Week Validation
- Week number must be unique within module
- Title must be provided
- Week number must be sequential (1-13)
- Learning objectives must be specified

### Topic Validation
- Title must be provided
- Content must be valid Markdown
- Topic must belong to exactly one week
- Difficulty level must be specified

### Page Validation
- Title must be provided
- Content must be valid Markdown/MDX
- Page must belong to exactly one topic

### Resource Validation
- Title must be provided
- URL must be valid
- Type must be specified
- Resource must belong to exactly one topic

### Exercise Validation
- Title must be provided
- Description must be provided
- Difficulty level must be specified
- Exercise must belong to exactly one topic