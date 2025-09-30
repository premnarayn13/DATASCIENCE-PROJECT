"""
Advanced Synthetic IT Support Ticket Data Generator

This module creates a comprehensive, realistic dataset of IT support tickets
with complex relationships, realistic patterns, and business logic constraints.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
import uuid
from pathlib import Path
import logging
from dataclasses import dataclass
from faker import Faker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

@dataclass
class TicketTemplate:
    """Template for generating realistic ticket descriptions"""
    category: str
    subcategory: str
    common_issues: List[str]
    technical_terms: List[str]
    user_descriptions: List[str]
    resolution_patterns: List[str]
    typical_duration_range: Tuple[int, int]  # hours

class SyntheticDataGenerator:
    """
    Advanced synthetic data generator for IT support tickets
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fake = Faker()
        
        # Initialize ticket templates
        self.ticket_templates = self._initialize_ticket_templates()
        
        # Initialize user profiles
        self.user_profiles = self._generate_user_profiles()
        
        # Initialize technician profiles
        self.technician_profiles = self._generate_technician_profiles()
        
        # Initialize system inventory
        self.system_inventory = self._generate_system_inventory()
        
        # Seasonal and business patterns
        self.seasonal_multipliers = self._get_seasonal_multipliers()
        self.business_hour_patterns = self._get_business_hour_patterns()
        
    def _initialize_ticket_templates(self) -> Dict[str, List[TicketTemplate]]:
        """Initialize realistic ticket templates for each category"""
        
        templates = {
            "Hardware": [
                TicketTemplate(
                    category="Hardware",
                    subcategory="Desktop Issues",
                    common_issues=[
                        "Computer won't start", "Blue screen error", "Slow performance",
                        "Hardware failure", "Power supply issues", "RAM problems"
                    ],
                    technical_terms=[
                        "BSOD", "POST", "RAM", "CPU", "GPU", "motherboard",
                        "power supply", "hard drive", "SSD", "thermal throttling"
                    ],
                    user_descriptions=[
                        "My computer is making weird noises and running very slow",
                        "The screen goes black randomly during work",
                        "Getting blue screen with error codes",
                        "Computer won't turn on this morning",
                        "Applications are crashing frequently"
                    ],
                    resolution_patterns=[
                        "Replaced faulty RAM module",
                        "Updated graphics drivers",
                        "Cleaned system thermal components",
                        "Replaced power supply unit",
                        "Performed hardware diagnostics"
                    ],
                    typical_duration_range=(2, 24)
                ),
                TicketTemplate(
                    category="Hardware",
                    subcategory="Printer Issues",
                    common_issues=[
                        "Printer offline", "Paper jam", "Poor print quality",
                        "Driver issues", "Network connectivity", "Toner replacement"
                    ],
                    technical_terms=[
                        "print spooler", "driver", "toner", "duplex", "PCL", "PostScript",
                        "network printer", "USB connection", "print queue"
                    ],
                    user_descriptions=[
                        "Printer shows offline but it's connected",
                        "Print quality is very poor with streaks",
                        "Paper keeps jamming in the printer",
                        "Can't find printer in the list",
                        "Prints are coming out blank"
                    ],
                    resolution_patterns=[
                        "Cleared print queue and restarted spooler",
                        "Updated printer drivers",
                        "Replaced toner cartridge",
                        "Fixed network connectivity issue",
                        "Cleared paper jam and calibrated printer"
                    ],
                    typical_duration_range=(1, 8)
                )
            ],
            "Software": [
                TicketTemplate(
                    category="Software",
                    subcategory="Application Issues",
                    common_issues=[
                        "Software won't launch", "Application crashes", "License issues",
                        "Update problems", "Performance issues", "Feature not working"
                    ],
                    technical_terms=[
                        "executable", "DLL", "registry", "license key", "update",
                        "patch", "compatibility", "dependencies", "installation"
                    ],
                    user_descriptions=[
                        "Excel crashes every time I open large files",
                        "Can't install the new software update",
                        "Application says license expired but it shouldn't be",
                        "Software is running extremely slow",
                        "Getting error messages when trying to save files"
                    ],
                    resolution_patterns=[
                        "Reinstalled application with latest version",
                        "Updated license configuration",
                        "Cleared application cache and preferences",
                        "Applied compatibility settings",
                        "Resolved dependency conflicts"
                    ],
                    typical_duration_range=(1, 12)
                )
            ],
            "Network": [
                TicketTemplate(
                    category="Network",
                    subcategory="Connectivity Issues",
                    common_issues=[
                        "No internet access", "Slow connection", "WiFi problems",
                        "VPN issues", "Email connectivity", "File sharing problems"
                    ],
                    technical_terms=[
                        "DNS", "DHCP", "IP address", "subnet", "gateway", "firewall",
                        "VPN", "port", "bandwidth", "latency", "packet loss"
                    ],
                    user_descriptions=[
                        "Internet is not working on my computer",
                        "Connection is very slow, can't load websites",
                        "Can't connect to company WiFi",
                        "VPN keeps disconnecting",
                        "Can't access shared network drives"
                    ],
                    resolution_patterns=[
                        "Reset network adapter and renewed IP",
                        "Updated network drivers",
                        "Reconfigured VPN settings",
                        "Resolved DNS configuration issue",
                        "Fixed firewall blocking rules"
                    ],
                    typical_duration_range=(1, 6)
                )
            ],
            "Security": [
                TicketTemplate(
                    category="Security",
                    subcategory="Access Management",
                    common_issues=[
                        "Password reset", "Account lockout", "Permission issues",
                        "Security software alerts", "Suspicious activity", "Two-factor authentication"
                    ],
                    technical_terms=[
                        "Active Directory", "LDAP", "SSO", "MFA", "2FA", "RBAC",
                        "permissions", "group policy", "authentication", "authorization"
                    ],
                    user_descriptions=[
                        "My password expired and I can't reset it",
                        "Account is locked out after multiple login attempts",
                        "Can't access files I had permission to yesterday",
                        "Antivirus is showing security alerts",
                        "Getting suspicious emails in my inbox"
                    ],
                    resolution_patterns=[
                        "Reset password and unlocked account",
                        "Updated security group permissions",
                        "Configured two-factor authentication",
                        "Removed malware and updated security software",
                        "Implemented additional security policies"
                    ],
                    typical_duration_range=(0.5, 4)
                )
            ],
            "Database": [
                TicketTemplate(
                    category="Database",
                    subcategory="Performance Issues",
                    common_issues=[
                        "Slow queries", "Database timeouts", "Connection issues",
                        "Data corruption", "Backup failures", "Storage issues"
                    ],
                    technical_terms=[
                        "SQL", "query optimization", "indexing", "connection pool",
                        "transaction log", "backup", "restore", "replication"
                    ],
                    user_descriptions=[
                        "Reports are taking forever to load",
                        "Database application keeps timing out",
                        "Can't connect to the main database",
                        "Data appears to be missing from reports",
                        "Getting database error messages"
                    ],
                    resolution_patterns=[
                        "Optimized database queries and rebuilt indexes",
                        "Increased connection pool size",
                        "Resolved database connectivity issues",
                        "Restored data from backup",
                        "Performed database maintenance"
                    ],
                    typical_duration_range=(2, 16)
                )
            ]
        }
        
        return templates
    
    def _generate_user_profiles(self) -> List[Dict[str, Any]]:
        """Generate realistic user profiles"""
        
        departments = self.config["departments"]
        roles = self.config["user_roles"]
        locations = self.config["locations"]
        
        users = []
        user_id = 1
        
        for dept in departments:
            # Number of users per department (realistic distribution)
            dept_size = {
                "IT": 25, "Finance": 40, "HR": 30, "Sales": 60,
                "Marketing": 35, "Operations": 50, "Executive": 10,
                "Legal": 15, "Procurement": 20
            }.get(dept, 30)
            
            for _ in range(dept_size):
                user = {
                    "user_id": f"USER_{user_id:04d}",
                    "name": self.fake.name(),
                    "email": self.fake.email(),
                    "department": dept,
                    "role": np.random.choice(roles, p=[0.05, 0.15, 0.25, 0.45, 0.05, 0.05]),
                    "location": np.random.choice(locations, p=[0.4, 0.3, 0.2, 0.1]),
                    "experience_level": np.random.choice(["Novice", "Intermediate", "Advanced"], p=[0.3, 0.5, 0.2]),
                    "tech_savviness": np.random.normal(0.5, 0.2),  # 0-1 scale
                    "avg_tickets_per_month": np.random.poisson(2.5),
                    "preferred_contact": np.random.choice(["email", "phone", "chat"], p=[0.6, 0.2, 0.2])
                }
                users.append(user)
                user_id += 1
        
        return users
    
    def _generate_technician_profiles(self) -> List[Dict[str, Any]]:
        """Generate realistic technician profiles"""
        
        specializations = [
            "Hardware", "Software", "Network", "Security", 
            "Database", "General Support"
        ]
        
        technicians = []
        tech_id = 1
        
        for spec in specializations:
            # Number of technicians per specialization
            team_size = np.random.randint(3, 8)
            
            for _ in range(team_size):
                tech = {
                    "technician_id": f"TECH_{tech_id:03d}",
                    "name": self.fake.name(),
                    "email": self.fake.email(),
                    "specialization": spec,
                    "experience_years": np.random.uniform(1, 15),
                    "skill_level": np.random.choice(["Junior", "Mid", "Senior"], p=[0.3, 0.5, 0.2]),
                    "max_concurrent_tickets": np.random.randint(5, 15),
                    "avg_resolution_time": np.random.uniform(2, 8),  # hours
                    "customer_satisfaction": np.random.uniform(3.5, 5.0),
                    "availability_schedule": self._generate_schedule(),
                    "workload_preference": np.random.choice(["light", "moderate", "heavy"], p=[0.2, 0.6, 0.2])
                }
                technicians.append(tech)
                tech_id += 1
        
        return technicians
    
    def _generate_schedule(self) -> Dict[str, List[int]]:
        """Generate weekly work schedule for technicians"""
        
        schedule = {}
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            if day in ["Saturday", "Sunday"]:
                # Weekend coverage (reduced)
                if np.random.random() < 0.3:  # 30% work weekends
                    schedule[day] = list(range(10, 18))  # 10 AM - 6 PM
                else:
                    schedule[day] = []
            else:
                # Weekday coverage
                start_hour = np.random.choice([8, 9])
                end_hour = np.random.choice([17, 18, 19])
                schedule[day] = list(range(start_hour, end_hour))
        
        return schedule
    
    def _generate_system_inventory(self) -> List[Dict[str, Any]]:
        """Generate IT system inventory"""
        
        systems = []
        
        # Generate various system types
        system_types = [
            {"type": "Server", "count": 50, "criticality": "High"},
            {"type": "Database", "count": 15, "criticality": "Critical"},
            {"type": "Network Device", "count": 100, "criticality": "High"},
            {"type": "Desktop", "count": 500, "criticality": "Medium"},
            {"type": "Laptop", "count": 300, "criticality": "Medium"},
            {"type": "Printer", "count": 80, "criticality": "Low"},
            {"type": "Phone System", "count": 20, "criticality": "Medium"},
            {"type": "Security Device", "count": 30, "criticality": "High"}
        ]
        
        system_id = 1
        for sys_type in system_types:
            for _ in range(sys_type["count"]):
                system = {
                    "system_id": f"SYS_{system_id:04d}",
                    "type": sys_type["type"],
                    "name": f"{sys_type['type']}_{system_id}",
                    "criticality": sys_type["criticality"],
                    "location": np.random.choice(self.config["locations"]),
                    "manufacturer": self._get_manufacturer(sys_type["type"]),
                    "model": self.fake.bothify("Model-####"),
                    "installation_date": self.fake.date_between(start_date="-5y", end_date="today"),
                    "last_maintenance": self.fake.date_between(start_date="-6m", end_date="today"),
                    "warranty_expiry": self.fake.date_between(start_date="today", end_date="+2y"),
                    "dependencies": self._generate_dependencies(system_id, sys_type["type"]),
                    "reliability_score": np.random.uniform(0.7, 0.99)
                }
                systems.append(system)
                system_id += 1
        
        return systems
    
    def _get_manufacturer(self, system_type: str) -> str:
        """Get realistic manufacturer based on system type"""
        
        manufacturers = {
            "Server": ["Dell", "HP", "Lenovo", "Cisco", "IBM"],
            "Desktop": ["Dell", "HP", "Lenovo", "Asus"],
            "Laptop": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
            "Printer": ["HP", "Canon", "Epson", "Brother", "Xerox"],
            "Network Device": ["Cisco", "Juniper", "Aruba", "Netgear"],
            "Database": ["Oracle", "Microsoft", "IBM", "MySQL"],
            "Phone System": ["Cisco", "Avaya", "Mitel", "RingCentral"],
            "Security Device": ["Fortinet", "Palo Alto", "Cisco", "SonicWall"]
        }
        
        return np.random.choice(manufacturers.get(system_type, ["Generic"]))
    
    def _generate_dependencies(self, system_id: int, system_type: str) -> List[str]:
        """Generate system dependencies"""
        
        # Simulate realistic dependencies
        if system_type == "Desktop":
            return [f"SYS_{np.random.randint(1, 51):04d}"]  # Depends on servers
        elif system_type == "Database":
            return [f"SYS_{np.random.randint(1, 51):04d}"]  # Depends on servers
        elif system_type in ["Printer", "Phone System"]:
            return [f"SYS_{np.random.randint(51, 151):04d}"]  # Depends on network devices
        else:
            return []
    
    def _get_seasonal_multipliers(self) -> Dict[int, float]:
        """Get seasonal multipliers for ticket volume"""
        
        return {
            1: 1.2,   # January - post-holiday issues
            2: 0.9,   # February - normal
            3: 1.1,   # March - quarter end
            4: 1.0,   # April - normal
            5: 0.95,  # May - normal
            6: 1.1,   # June - quarter end
            7: 0.8,   # July - summer vacation
            8: 0.85,  # August - summer vacation
            9: 1.3,   # September - back to work
            10: 1.1,  # October - normal
            11: 0.9,  # November - normal
            12: 1.2   # December - year-end issues
        }
    
    def _get_business_hour_patterns(self) -> Dict[int, float]:
        """Get hourly patterns for ticket submission"""
        
        patterns = {}
        for hour in range(24):
            if 6 <= hour <= 8:    # Early morning
                patterns[hour] = 0.3
            elif 9 <= hour <= 11:  # Morning peak
                patterns[hour] = 1.5
            elif 12 <= hour <= 13: # Lunch
                patterns[hour] = 0.8
            elif 14 <= hour <= 16: # Afternoon peak
                patterns[hour] = 1.3
            elif 17 <= hour <= 18: # End of day
                patterns[hour] = 1.0
            elif 19 <= hour <= 22: # Evening
                patterns[hour] = 0.4
            else:                  # Night
                patterns[hour] = 0.1
        
        return patterns
    
    def generate_ticket_description(self, template: TicketTemplate, 
                                  user_profile: Dict[str, Any]) -> Tuple[str, str]:
        """Generate realistic ticket description and resolution"""
        
        # Choose issue based on user tech savviness
        tech_level = user_profile["tech_savviness"]
        
        if tech_level < 0.3:  # Low tech - more basic descriptions
            issue = np.random.choice(template.user_descriptions)
            description = f"{issue}. Please help as soon as possible."
        elif tech_level > 0.7:  # High tech - more technical descriptions
            issue = np.random.choice(template.common_issues)
            technical_detail = np.random.choice(template.technical_terms)
            description = f"Experiencing {issue}. Issue seems related to {technical_detail}. " \
                         f"Attempted basic troubleshooting but need technical assistance."
        else:  # Medium tech
            issue = np.random.choice(template.user_descriptions)
            description = f"{issue}. I've tried restarting but the problem persists."
        
        # Add urgency based on user role and department
        if user_profile["role"] in ["Executive", "Manager"]:
            description += " This is impacting my work and needs urgent attention."
        elif user_profile["department"] in ["Finance", "Sales"]:
            description += " This is affecting customer service."
        
        # Generate resolution
        resolution = np.random.choice(template.resolution_patterns)
        if np.random.random() < 0.3:  # 30% chance of additional steps
            additional_step = "Also provided user training to prevent similar issues."
            resolution += f" {additional_step}"
        
        return description, resolution
    
    def calculate_resolution_time(self, template: TicketTemplate, 
                                ticket_data: Dict[str, Any]) -> float:
        """Calculate realistic resolution time based on multiple factors"""
        
        base_time = np.random.uniform(*template.typical_duration_range)
        
        # Adjust based on priority
        priority_multipliers = {
            "Emergency": 0.3,
            "Critical": 0.5,
            "High": 0.8,
            "Medium": 1.0,
            "Low": 1.5
        }
        base_time *= priority_multipliers[ticket_data["priority"]]
        
        # Adjust based on user department (some departments get faster service)
        dept_multipliers = {
            "Executive": 0.6,
            "Finance": 0.8,
            "IT": 0.7,
            "Sales": 0.9,
            "HR": 1.0,
            "Marketing": 1.1,
            "Operations": 1.0,
            "Legal": 0.9,
            "Procurement": 1.2
        }
        base_time *= dept_multipliers.get(ticket_data["department"], 1.0)
        
        # Add some randomness
        base_time *= np.random.uniform(0.7, 1.4)
        
        # Ensure minimum resolution time
        return max(0.5, base_time)
    
    def assign_priority(self, template: TicketTemplate, 
                       user_profile: Dict[str, Any]) -> str:
        """Assign priority based on issue type and user profile"""
        
        # Base priority probabilities for each category
        category_priorities = {
            "Hardware": {"Emergency": 0.05, "Critical": 0.15, "High": 0.30, "Medium": 0.40, "Low": 0.10},
            "Software": {"Emergency": 0.02, "Critical": 0.08, "High": 0.25, "Medium": 0.50, "Low": 0.15},
            "Network": {"Emergency": 0.10, "Critical": 0.20, "High": 0.35, "Medium": 0.25, "Low": 0.10},
            "Security": {"Emergency": 0.15, "Critical": 0.25, "High": 0.30, "Medium": 0.20, "Low": 0.10},
            "Database": {"Emergency": 0.08, "Critical": 0.22, "High": 0.40, "Medium": 0.25, "Low": 0.05},
            "Email": {"Emergency": 0.03, "Critical": 0.12, "High": 0.30, "Medium": 0.45, "Low": 0.10},
            "Access Management": {"Emergency": 0.05, "Critical": 0.15, "High": 0.35, "Medium": 0.35, "Low": 0.10},
            "Infrastructure": {"Emergency": 0.12, "Critical": 0.28, "High": 0.35, "Medium": 0.20, "Low": 0.05}
        }
        
        priorities = list(category_priorities[template.category].keys())
        probabilities = list(category_priorities[template.category].values())
        
        # Adjust probabilities based on user role
        if user_profile["role"] in ["Executive", "Manager"]:
            # Increase high priority probabilities
            probabilities = [p * 1.5 if p in probabilities[:3] else p * 0.7 for p in probabilities]
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return np.random.choice(priorities, p=probabilities)
    
    def generate_tickets(self, num_tickets: int = None) -> pd.DataFrame:
        """Generate the complete synthetic ticket dataset"""
        
        if num_tickets is None:
            num_tickets = self.config["num_tickets"]
        
        logger.info(f"Generating {num_tickets} synthetic tickets...")
        
        tickets = []
        
        # Date range for tickets
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.config["date_range_years"])
        
        for ticket_id in range(1, num_tickets + 1):
            # Generate random timestamp with realistic patterns
            random_date = self.fake.date_time_between(start_date=start_date, end_date=end_date)
            
            # Apply seasonal and business hour patterns
            hour_multiplier = self.business_hour_patterns[random_date.hour]
            month_multiplier = self.seasonal_multipliers[random_date.month]
            
            # Skip some tickets based on patterns (simulate realistic submission times)
            if np.random.random() > (hour_multiplier * month_multiplier / 2):
                if np.random.random() < 0.7:  # 70% chance to skip
                    continue
            
            # Select random category and template
            category = np.random.choice(list(self.ticket_templates.keys()))
            template = np.random.choice(self.ticket_templates[category])
            
            # Select random user
            user = np.random.choice(self.user_profiles)
            
            # Generate ticket data
            priority = self.assign_priority(template, user)
            description, resolution_notes = self.generate_ticket_description(template, user)
            
            # Create ticket data
            ticket_data = {
                "ticket_id": f"TKT_{ticket_id:06d}",
                "created_at": random_date,
                "category": template.category,
                "subcategory": template.subcategory,
                "priority": priority,
                "description": description,
                "user_id": user["user_id"],
                "user_name": user["name"],
                "user_email": user["email"],
                "department": user["department"],
                "user_role": user["role"],
                "user_location": user["location"],
                "affected_system": np.random.choice(self.system_inventory)["system_id"] if np.random.random() < 0.7 else None,
                "resolution_notes": resolution_notes
            }
            
            # Calculate resolution time
            resolution_hours = self.calculate_resolution_time(template, ticket_data)
            resolution_time = random_date + timedelta(hours=resolution_hours)
            
            ticket_data.update({
                "resolved_at": resolution_time,
                "resolution_time_hours": resolution_hours,
                "status": "Resolved"
            })
            
            # Assign technician
            suitable_techs = [t for t in self.technician_profiles 
                            if t["specialization"] == template.category or t["specialization"] == "General Support"]
            if suitable_techs:
                technician = np.random.choice(suitable_techs)
                ticket_data.update({
                    "assigned_technician": technician["technician_id"],
                    "technician_name": technician["name"],
                    "technician_specialization": technician["specialization"]
                })
            
            # Add escalation information
            if priority in ["Emergency", "Critical"] and np.random.random() < 0.3:
                ticket_data["escalated"] = True
                ticket_data["escalation_reason"] = np.random.choice([
                    "SLA breach", "Customer escalation", "Technical complexity", "Resource unavailability"
                ])
            else:
                ticket_data["escalated"] = False
                ticket_data["escalation_reason"] = None
            
            # Add customer satisfaction score
            satisfaction_base = np.random.uniform(3.0, 5.0)
            if resolution_hours <= template.typical_duration_range[0]:
                satisfaction_base += 0.5  # Bonus for quick resolution
            elif resolution_hours >= template.typical_duration_range[1] * 1.5:
                satisfaction_base -= 1.0  # Penalty for slow resolution
            
            ticket_data["customer_satisfaction"] = np.clip(satisfaction_base, 1.0, 5.0)
            
            # Add reopened flag
            ticket_data["reopened"] = np.random.random() < 0.05  # 5% reopening rate
            if ticket_data["reopened"]:
                ticket_data["reopen_count"] = np.random.randint(1, 4)
            else:
                ticket_data["reopen_count"] = 0
            
            tickets.append(ticket_data)
            
            if len(tickets) % 1000 == 0:
                logger.info(f"Generated {len(tickets)} tickets...")
        
        logger.info(f"Successfully generated {len(tickets)} tickets")
        
        # Convert to DataFrame
        df = pd.DataFrame(tickets)
        
        # Add calculated features
        df = self._add_calculated_features(df)
        
        return df
    
    def _add_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features for ML"""
        
        # Time-based features
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        
        df['hour_of_day'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['month'] = df['created_at'].dt.month
        df['quarter'] = df['created_at'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour_of_day'].between(9, 17)
        
        # User-based aggregations
        user_stats = df.groupby('user_id').agg({
            'ticket_id': 'count',
            'resolution_time_hours': ['mean', 'std'],
            'customer_satisfaction': 'mean',
            'escalated': 'mean'
        }).round(2)
        
        user_stats.columns = ['user_ticket_count', 'user_avg_resolution_time', 
                             'user_resolution_time_std', 'user_avg_satisfaction', 'user_escalation_rate']
        
        df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # Department-based aggregations
        dept_stats = df.groupby('department').agg({
            'resolution_time_hours': 'mean',
            'customer_satisfaction': 'mean',
            'escalated': 'mean'
        }).round(2)
        
        dept_stats.columns = ['dept_avg_resolution_time', 'dept_avg_satisfaction', 'dept_escalation_rate']
        df = df.merge(dept_stats, left_on='department', right_index=True, how='left')
        
        # Category-based aggregations
        cat_stats = df.groupby('category').agg({
            'resolution_time_hours': 'mean',
            'escalated': 'mean'
        }).round(2)
        
        cat_stats.columns = ['category_avg_resolution_time', 'category_escalation_rate']
        df = df.merge(cat_stats, left_on='category', right_index=True, how='left')
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save the generated dataset"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        logger.info(f"Saved main dataset to {output_path}")
        
        # Save metadata
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_tickets": len(df),
            "date_range": {
                "start": df['created_at'].min().isoformat(),
                "end": df['created_at'].max().isoformat()
            },
            "categories": df['category'].value_counts().to_dict(),
            "priorities": df['priority'].value_counts().to_dict(),
            "departments": df['department'].value_counts().to_dict(),
            "statistics": {
                "avg_resolution_time": df['resolution_time_hours'].mean(),
                "median_resolution_time": df['resolution_time_hours'].median(),
                "avg_customer_satisfaction": df['customer_satisfaction'].mean(),
                "escalation_rate": df['escalated'].mean(),
                "reopen_rate": df['reopened'].mean()
            }
        }
        
        metadata_path = output_path.parent / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save supporting data
        self._save_supporting_data(output_path.parent)
    
    def _save_supporting_data(self, output_dir: Path):
        """Save supporting datasets"""
        
        # Save user profiles
        user_df = pd.DataFrame(self.user_profiles)
        user_df.to_csv(output_dir / "user_profiles.csv", index=False)
        
        # Save technician profiles
        tech_df = pd.DataFrame(self.technician_profiles)
        tech_df.to_csv(output_dir / "technician_profiles.csv", index=False)
        
        # Save system inventory
        system_df = pd.DataFrame(self.system_inventory)
        system_df.to_csv(output_dir / "system_inventory.csv", index=False)
        
        logger.info("Saved supporting datasets")

def main():
    """Main function to generate synthetic data"""
    
    # Import configuration
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.settings import DATA_CONFIG
    
    # Initialize generator
    generator = SyntheticDataGenerator(DATA_CONFIG["synthetic_data"])
    
    # Generate dataset
    tickets_df = generator.generate_tickets()
    
    # Save dataset
    output_path = Path(__file__).parent.parent.parent / "data" / "synthetic_tickets.csv"
    generator.save_dataset(tickets_df, output_path)
    
    print(f"\n📊 Dataset Generation Complete!")
    print(f"Generated {len(tickets_df)} tickets")
    print(f"Date range: {tickets_df['created_at'].min()} to {tickets_df['created_at'].max()}")
    print(f"Average resolution time: {tickets_df['resolution_time_hours'].mean():.2f} hours")
    print(f"Customer satisfaction: {tickets_df['customer_satisfaction'].mean():.2f}/5.0")

if __name__ == "__main__":
    main()